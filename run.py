"""Plan (G-FOLD + SCvx), simulate all configured cases, and export data for the 3D viewer.

    uv run python run.py [configs/scenarios/default.json] [--out web/data/results.json] [--serve]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from landing.dynamics import TruthModel, open_loop_resim
from landing.guidance import GFoldGuidance, SCvxGuidance
from landing.ignition import plan_with_ignition
from landing.sim import simulate, truth_from_case
from landing.vehicle import Scenario

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="configs/scenarios/coast_landing.json")
    ap.add_argument("--out", default="web/data/results.json")
    ap.add_argument("--serve", action="store_true", help="serve web/ and open the viewer")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    sc = Scenario.from_json(args.scenario)
    rk = sc.rocket
    print(f"Scenario: {sc.name} | rocket: {rk.name}")

    guidance = {"gfold": GFoldGuidance(sc), "scvx": SCvxGuidance(sc)}

    def make_plan(key: str, **kw):
        if sc.coast:  # engine-off start: choose ignition time, then plan from the predicted state
            return plan_with_ignition(sc, guidance[key], verbose=verbose)
        return guidance[key].plan(verbose=verbose, **kw)

    t0 = time.perf_counter()
    plan_g = make_plan("gfold")
    plan_g.info["solve_time_s"] = time.perf_counter() - t0
    resim = open_loop_resim(TruthModel(rk, sc.env, drag=False), plan_g)
    plan_g.info["open_loop_terminal_err"] = [float(np.linalg.norm(resim[0:3])), float(np.linalg.norm(resim[3:6]))]

    t0 = time.perf_counter()
    plan_s = make_plan("scvx", **({} if sc.coast else {"init": plan_g}))
    plan_s.info["solve_time_s"] = time.perf_counter() - t0
    resim = open_loop_resim(TruthModel(rk, sc.env, drag=True), plan_s)
    plan_s.info["open_loop_terminal_err"] = [float(np.linalg.norm(resim[0:3])), float(np.linalg.norm(resim[3:6]))]

    plans = {"gfold": plan_g, "scvx": plan_s}
    print("\nPlans")
    print(f"  {'':8s} {'ignite':>7s} {'h_ig':>7s} {'burn':>7s} {'m_f [kg]':>10s} {'fuel [kg]':>10s} {'tight':>9s} {'resim r/v err':>16s}")
    for k, p in plans.items():
        e = p.info["open_loop_terminal_err"]
        print(
            f"  {k:8s} {p.info.get('t_ignition', 0.0):6.2f}s {p.r[0, 0]:6.0f}m {p.tf:6.2f}s {p.m[-1]:10.1f} {sc.m0 - p.m[-1]:10.1f} "
            f"{p.info['tightness_max']:9.1e} {e[0]:7.4f}m {e[1]:6.4f}m/s"
        )

    sims = []
    print("\nClosed-loop simulations")
    print(f"  {'case':62s} {'pos err':>8s} {'v_vert':>7s} {'v_hor':>6s} {'tilt':>5s} {'fuel':>7s} {'gsat%':>5s} {'replans':>7s}")
    for case in sc.sim_cases:
        t0 = time.perf_counter()
        res = simulate(
            sc,
            plans[case["plan"]],
            truth_from_case(sc, case.get("truth", {})),
            case["name"],
            guidance=guidance[case["plan"]],
            replan_hz=case.get("replan_hz", 0.0),
            controller=case.get("controller", "cascade"),
        )
        res.metrics["wall_time_s"] = time.perf_counter() - t0
        res.metrics["plan"] = case["plan"]
        res.metrics["truth"] = case.get("truth", {})
        mt = res.metrics
        print(
            f"  {case['name']:62s} {mt['touchdown_pos_err_m']:7.2f}m {mt['touchdown_vert_speed']:6.2f} "
            f"{mt['touchdown_horiz_speed']:6.2f} {mt['touchdown_tilt_deg']:5.1f} {mt['fuel_used_kg']:7.0f} {100 * mt['gimbal_sat_frac']:5.1f} "
            f"{mt['n_replans']:4d}/{mt['n_replan_failures']}"
        )
        sims.append(res.to_dict(rk.T_max))

    data = {
        "scenario": {
            "name": sc.name,
            "m0": sc.m0,
            "r0": sc.r0.tolist(),
            "v0": sc.v0.tolist(),
            "glide_slope_deg": sc.glide_slope_deg,
            "wind": list(sc.env.wind),
        },
        "rocket": {
            "name": rk.name, "m_dry": rk.m_dry, "T_max": rk.T_max, "thr_min": rk.thr_min,
            "thr_max": rk.thr_max, "theta_max_deg": rk.theta_max_deg, "isp": rk.isp,
            "S_ref": rk.S_ref, "C_D": rk.C_D, "geometry": rk.geometry,
        },
        "plans": {k: p.to_dict(rk.T_max) for k, p in plans.items()},
        "sims": sims,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, default=float, ensure_ascii=False))
    print(f"\nWrote {out}")
    if args.serve:
        serve(args.port, out)
    else:
        print("View it with:  uv run python run.py --serve   (or: uv run python -m http.server -d web)")


def serve(port: int, data_path: Path) -> None:
    import functools
    import http.server
    import webbrowser

    web = Path(__file__).resolve().parent / "web"
    rel = data_path.resolve().relative_to(web).as_posix()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/index.html?data={rel}"
        print(f"Serving viewer at {url}  (Ctrl+C to stop)")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
