"""Closed-loop simulator: truth dynamics + reference-tracking controller (spec §6, simplified).

The guidance plan is computed once at t=0; the tracker (~100 Hz) follows it:
    a_cmd = u_ref + Kr (r_ref - r) + Kv (v_ref - v),   T_cmd = m * a_cmd -> projected onto
    the physical thrust set (magnitude in [rho1, T_max], within the pointing cone).
At t >= t_f the engine is cut (hoverslam: it cannot hover since min thrust > weight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .attitude import AttitudeModel, AttitudeParams, AttitudeState, ReferenceModel
from .dynamics import TruthModel
from .mpc import TrackingMPC, embed_gimbal
from .trajectory import Trajectory
from .vehicle import E1, Scenario, rotate_toward


def max_rate(t: np.ndarray, att: np.ndarray) -> float:
    dots = np.clip(np.sum(att[1:] * att[:-1], axis=1), -1, 1)
    dt = np.diff(t)
    ok = dt > 1e-6
    return float(np.max(np.arccos(dots[ok]) / dt[ok])) if ok.any() else 0.0


def project_thrust(T: np.ndarray, t_min: float, t_max: float, theta_max: float) -> np.ndarray:
    """Project a commanded thrust onto the pointing cone (Euclidean projection), then clip the
    magnitude to [t_min, t_max]. Downward commands (projection = apex) map to min vertical thrust."""
    n = np.linalg.norm(T)
    if n < 1e-9:
        return t_min * E1
    d = T / n
    mag = n
    if d[0] < math.cos(theta_max):
        hor = d.copy()
        hor[0] = 0.0
        hn = np.linalg.norm(hor)
        hor = hor / hn if hn > 1e-12 else np.array([0.0, 1.0, 0.0])
        d = math.cos(theta_max) * E1 + math.sin(theta_max) * hor
        mag = float(T @ d)  # length of the projection onto the cone edge
        if mag <= 0.0:
            return t_min * E1
    return d * float(np.clip(mag, t_min, t_max))


@dataclass
class SimResult:
    name: str
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    m: np.ndarray
    T: np.ndarray
    r_ref: np.ndarray
    att: np.ndarray  # body axis (unit vector, guidance frame)
    gimbal: np.ndarray  # engine gimbal vector (perp. to att, |g| = sin(deflection), guidance frame)
    metrics: dict[str, Any] = field(default_factory=dict)
    replans: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, T_max: float, decimate: int = 2) -> dict[str, Any]:
        idx = np.r_[np.arange(0, len(self.t) - 1, decimate), len(self.t) - 1]
        thrust = np.linalg.norm(self.T, axis=1)
        return {
            "name": self.name,
            "t": self.t[idx].round(4).tolist(),
            "r": self.r[idx].round(4).tolist(),
            "v": self.v[idx].round(4).tolist(),
            "m": self.m[idx].round(2).tolist(),
            "T": self.T[idx].round(1).tolist(),
            "throttle": (thrust[idx] / T_max).round(4).tolist(),
            "r_ref": self.r_ref[idx].round(4).tolist(),
            "att": self.att[idx].round(5).tolist(),
            "gimbal": self.gimbal[idx].round(5).tolist(),
            "metrics": self.metrics,
            "replans": self.replans,
        }


def simulate(
    sc: Scenario,
    plan: Trajectory,
    truth: TruthModel,
    name: str = "",
    guidance=None,
    replan_hz: float = 0.0,
    freeze_tgo: float = 2.0,
    probe=None,
    controller: str = "cascade",
) -> SimResult:
    """Fly `plan` through `truth`.

    If the plan carries `info["t_ignition"]` the vehicle first coasts engine-off until then.
    If `guidance` and `replan_hz` > 0, re-plan from the current (perfectly known) state at
    ignition and then at that rate until t_go < freeze_tgo, then hold the last plan.
    `probe(signals: dict)` is called every powered control step with every intermediate signal.
    `controller`: "cascade" (PD tracker -> reference model -> attitude loop -> gimbal) or
    "mpc" (attitude-aware tracking MPC commanding throttle and gimbal directly, landing/mpc.py)."""
    rk = sc.rocket
    cfg = {"omega_n": 1.0, "zeta": 0.7, "dt": 0.01, **sc.tracker}
    dt = cfg["dt"]
    # vertical channel acts through throttle (fast); lateral channel through body rotation (slow)
    # -> separate gains so the outer loop stays well below the attitude-loop bandwidth
    wl, zl = cfg.get("omega_n_lat", cfg["omega_n"]), cfg.get("zeta_lat", cfg["zeta"])
    Kr = np.array([cfg["omega_n"] ** 2, wl**2, wl**2])
    Kv = np.array([2 * cfg["zeta"] * cfg["omega_n"], 2 * zl * wl, 2 * zl * wl])

    t_ig = float(plan.info.get("t_ignition", 0.0))
    active, plan_t0 = plan, t_ig
    replan_dt = 1.0 / replan_hz if (guidance is not None and replan_hz > 0) else math.inf
    next_replan = (t_ig if t_ig > 0 else replan_dt) if math.isfinite(replan_dt) else math.inf
    replans: list[dict[str, Any]] = []
    n_replan_fail = 0

    # attitude: rigid-body pitch/yaw driven by the engine gimbal (see attitude.py). The tracker's
    # thrust command sets the desired body axis; thrust acts along the deflected nozzle.
    # Engine off -> grid fins keep the body aligned with the airflow (falling engine-first).
    wind = truth.wind if truth.wind is not None else sc.env.wind_vec
    att_model = AttitudeModel(AttitudeParams.from_dict(rk.attitude))

    def airflow_axis(v):
        vr = v - wind
        n = np.linalg.norm(vr)
        return -vr / n if n > 1e-6 else E1.copy()

    s = np.concatenate([sc.r0, sc.v0, [sc.m0]])
    b0 = airflow_axis(sc.v0) if sc.att0 is None else sc.att0 / np.linalg.norm(sc.att0)
    ast = AttitudeState(b=b0)
    # smooth pointing reference (bounded rate and angular acceleration) for the attitude loop
    ref = ReferenceModel(b0, omega=cfg.get("ref_omega", 1.5), rate_max=math.radians(cfg.get("cmd_rate_deg_s", 8.0)))
    ref_brake = cfg.get("ref_accel_frac", 0.5)  # fraction of gimbal authority the reference may use
    lat_fade = float(cfg.get("lateral_fade_s", 0.0))
    t = 0.0
    log_t, log_s, log_T, log_ref, log_att, log_gim = [t], [s.copy()], [], [], [], []
    max_gimbal = 0.0
    steps = 0
    max_track_err = 0.0
    gimbal_sat_steps = 0
    use_mpc = controller == "mpc"
    if use_mpc:
        mpc = TrackingMPC(sc, drag=truth.drag)
        sol, sol_t0, next_mpc = None, 0.0, -math.inf
        mpc_solves, mpc_fail, mpc_time = 0, 0, 0.0
    elif controller != "cascade":
        raise ValueError(f"unknown controller {controller!r}")

    while True:
        r, v, m = s[0:3], s[3:6], s[6]
        tau = t - plan_t0
        t_go = active.tf - tau

        if t < t_ig - 1e-9:  # engine-off coast before ignition
            T = np.zeros(3)
            h = min(dt, t_ig - t)
            ast = att_model.step(ast, airflow_axis(v), 0.0, h)
            ref.reset(ast.b, ast.w)
            log_T.append(T)
            log_ref.append(r.copy())
            log_att.append(ast.b.copy())
            log_gim.append(np.zeros(3))
            s = truth.step(s, T, h)
            t += h
            log_t.append(t)
            log_s.append(s.copy())
            if s[0] <= 0.0:
                break
            continue

        if t >= next_replan - 1e-9:
            next_replan += replan_dt
            if t_go > freeze_tgo:
                new = guidance.replan(r.copy(), v.copy(), m, active, tau, att=ast.b.copy())
                if new is None:
                    n_replan_fail += 1
                else:
                    active, plan_t0, tau = new, t, 0.0
                    t_go = active.tf
                    if use_mpc:
                        next_mpc = -math.inf
                    replans.append({"t0": t, "tf": active.tf, "r": active.r.round(3).tolist(),
                                    "relaxed": bool(active.info.get("relaxed_attitude", False))})

        if t_go > 1e-9 and use_mpc:
            r_ref, v_ref = active.state_at(tau)
            h = min(dt, t_go)
            if t >= next_mpc - 1e-9 and (sol is None or t_go > mpc.p.stop_tgo):
                next_mpc = t + mpc.p.dt0
                new_sol = mpc.solve(active, tau, r, v, m, ast)
                mpc_solves += 1
                if new_sol is None:
                    mpc_fail += 1
                else:
                    sol, sol_t0 = new_sol, t
                    mpc_time += new_sol.info["solve_s"]
            if sol is None:
                raise RuntimeError("MPC failed on its first solve")
            Tn = sol.command(t + 0.5 * h - sol_t0)[0]  # thrust held over the interval
            g_h = sol.command(t + h - sol_t0)[1]  # gimbal target at the end of this step
            Tn = float(np.clip(Tn, rk.T_min, rk.T_max)) if m > rk.m_dry else 0.0
            ast = att_model.step_gimbal(ast, embed_gimbal(g_h, ast.b), Tn, h)
            T = Tn * ast.thrust_dir
            max_gimbal = max(max_gimbal, ast.gimbal_deg)
            if probe is not None:
                probe({
                    "t": t, "tau": tau, "t_go": t_go, "r": r.copy(), "v": v.copy(), "m": m,
                    "r_ref": r_ref, "v_ref": v_ref, "Tn": Tn, "g_h_cmd": g_h,
                    **{k: np.array(v_) for k, v_ in att_model.last.items()},
                    "b": ast.b.copy(), "w": ast.w.copy(), "t_hat": ast.thrust_dir, "T": T.copy(),
                    "mpc_x": sol.x, "mpc_t": sol.t + sol_t0, "plan": active,
                })
            max_track_err = max(max_track_err, float(np.linalg.norm(r_ref - r)))
        elif t_go > 1e-9:
            r_ref, v_ref = active.state_at(tau)
            u_ref = active.u_at(tau)
            # lateral corrections fade out over the last `lateral_fade_s`: near touchdown the
            # vehicle follows the (upright) plan instead of tilting to chase small position errors
            fade = np.array([1.0, *[min(1.0, t_go / lat_fade)] * 2]) if lat_fade > 0 else 1.0
            a_cmd = u_ref + fade * (Kr * (r_ref - r) + Kv * (v_ref - v))
            T_cmd = m * a_cmd
            theta = rk.theta_max
            T = project_thrust(T_cmd, rk.T_min, rk.T_max, theta)
            h = min(dt, t_go)
            # magnitude from the *current* nozzle direction: vertical-priority, so a lagging
            # body tilt is compensated by throttle (clipped to engine limits)
            td = ast.thrust_dir
            Tn = float(np.linalg.norm(T))
            if td[0] > 0.2:
                Tn = float(np.clip(m * a_cmd[0] / td[0], rk.T_min, rk.T_max))
            if m <= rk.m_dry:
                Tn = 0.0
            # attitude: smooth the commanded direction (reference model), then the attitude loop
            # drives the gimbal; the actual thrust acts along the nozzle (body axis + gimbal)
            ref.step(T / np.linalg.norm(T), h, ref_brake * att_model.alpha_avail(max(Tn, rk.T_min)))
            ast = att_model.step(ast, ref.b, Tn, h, w_ff=ref.w, a_ff=ref.a)
            if probe is not None:
                T_dir_cmd = T / np.linalg.norm(T)
            T = Tn * ast.thrust_dir
            max_gimbal = max(max_gimbal, ast.gimbal_deg)
            if probe is not None:
                probe({
                    "t": t, "tau": tau, "t_go": t_go, "r": r.copy(), "v": v.copy(), "m": m,
                    "r_ref": r_ref, "v_ref": v_ref, "u_ref": u_ref, "a_cmd": a_cmd,
                    "T_cmd": T_cmd, "dir_cmd": T_dir_cmd, "Tn": Tn,
                    "b_ref": ref.b.copy(), "w_ref": ref.w.copy(), "a_ref": ref.a.copy(),
                    **{k: np.array(v_) for k, v_ in att_model.last.items()},
                    "b": ast.b.copy(), "w": ast.w.copy(), "t_hat": ast.thrust_dir, "T": T.copy(),
                    "plan_N": len(active.t), "plan_tf": active.tf, "plan_hold": active.hold,
                })
            max_track_err = max(max_track_err, float(np.linalg.norm(r_ref - r)))
        else:  # engine cut at end of plan: ballistic until contact (fins hold attitude)
            r_ref, T = np.zeros(3), np.zeros(3)
            h = dt
            ast = att_model.step(ast, ast.b, 0.0, h)
        log_T.append(T)
        log_ref.append(r_ref)
        log_att.append(ast.b.copy())
        log_gim.append(ast.g.copy() if np.linalg.norm(T) > 0 else np.zeros(3))
        if np.linalg.norm(T) > 0 and ast.gimbal_deg >= 0.98 * att_model.p.gimbal_max_deg:
            gimbal_sat_steps += 1

        s_new = truth.step(s, T, h)
        t_new = t + h
        steps += 1
        if s_new[0] <= 0.0:  # ground contact: interpolate to h = 0
            frac = s[0] / (s[0] - s_new[0]) if s[0] > 0 else 0.0
            s_new = s + frac * (s_new - s)
            t_new = t + frac * h
            log_t.append(t_new)
            log_s.append(s_new)
            break
        s, t = s_new, t_new
        log_t.append(t)
        log_s.append(s.copy())
        if t > plan_t0 + active.tf + 30.0:  # never came down -> give up
            break

    log_T.append(log_T[-1])
    log_ref.append(log_ref[-1])
    log_att.append(ast.b.copy())
    log_gim.append(log_gim[-1] if log_gim else np.zeros(3))
    S = np.array(log_s)
    res = SimResult(
        name=name,
        t=np.array(log_t),
        r=S[:, 0:3],
        v=S[:, 3:6],
        m=S[:, 6],
        T=np.array(log_T),
        r_ref=np.array(log_ref),
        att=np.array(log_att),
        gimbal=np.array(log_gim),
    )
    v_td = S[-1, 3:6]
    res.metrics = {
        "landed": bool(S[-1, 0] <= 1e-6),
        "touchdown_time": float(log_t[-1]),
        "plan_tf": plan.tf,
        "t_ignition": t_ig,
        "touchdown_pos_err_m": float(np.linalg.norm(S[-1, 1:3])),
        "touchdown_vert_speed": float(-v_td[0]),
        "touchdown_horiz_speed": float(np.linalg.norm(v_td[1:3])),
        "touchdown_tilt_deg": float(np.degrees(np.arccos(np.clip(ast.b[0], -1, 1)))),
        "touchdown_body_rate_deg_s": float(np.degrees(np.linalg.norm(ast.w))),
        "max_slew_rate_deg_s": float(np.degrees(max_rate(np.array(log_t), np.array(log_att)))),
        "max_gimbal_deg": max_gimbal,
        "fuel_used_kg": float(sc.m0 - S[-1, 6]),
        "fuel_margin_kg": float(S[-1, 6] - rk.m_dry),
        "max_tracking_err_m": max_track_err,
        "gimbal_sat_frac": gimbal_sat_steps / max(steps, 1),
        "controller": controller,
        "replan_hz": replan_hz,
        "n_replans": len(replans),
        "n_replan_failures": n_replan_fail,
        "n_replans_relaxed": sum(rp["relaxed"] for rp in replans),
    }
    if use_mpc:
        res.metrics.update(mpc_solves=mpc_solves, mpc_failures=mpc_fail,
                           mpc_mean_solve_ms=1e3 * mpc_time / max(mpc_solves - mpc_fail, 1))
    res.replans = replans
    return res


def truth_from_case(sc: Scenario, case_truth: dict[str, Any]) -> TruthModel:
    return TruthModel(
        rocket=sc.rocket,
        env=sc.env,
        drag=case_truth.get("drag", True),
        cd_scale=case_truth.get("cd_scale", 1.0),
        isp_scale=case_truth.get("isp_scale", 1.0),
        wind=np.asarray(case_truth["wind"], float) if "wind" in case_truth else None,
    )
