import { TimeChart } from "./charts.js";
import { gimbalAt, gimbalPY, pitchYaw } from "./util.js";

const PLAN_LABEL = { gfold: "G-FOLD (no drag model)", scvx: "SCvx (drag model)" };
const f = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : typeof v === "number" ? v.toFixed(d) : String(v));
const row = (k, v, u = "") => `<div class="kv"><span>${k}</span><b>${v}</b>${u ? `<i>${u}</i>` : ""}</div>`;
const card = (title, body, cls = "") => `<section class="dcard ${cls}"><h3>${title}</h3>${body}</section>`;

/** Full report for one run: configuration, outcome, guidance plan, control effort, time histories. */
export function renderDetails(el, c, { rocket, scenario, verdict, describe }) {
  const sim = c.sim, m = sim.metrics, p = c.planData, pi = p?.info ?? {};
  const tr = m.truth || {};
  const [cls, label] = verdict(m);
  const vtd = Math.hypot(m.touchdown_vert_speed, m.touchdown_horiz_speed);
  const fuelPlan = p ? scenario.m0 - p.m.at(-1) : null;
  const solved = sim.solved_at ?? null;

  const config = [
    row("Guidance", PLAN_LABEL[m.plan] ?? m.plan),
    row("Controller", c.ctrl),
    row("Re-planning", m.replan_hz ? `${m.replan_hz} Hz` : "off (plan fixed)"),
    row("Truth drag", tr.drag === false ? "off (vacuum)" : "on"),
    row("C_D scale", f(tr.cd_scale ?? 1, 2), "×"),
    row("Wind", tr.wind ? `[${tr.wind.map((w) => w.toFixed(1)).join(", ")}]` : "none", tr.wind ? "m/s" : ""),
    row("Isp scale", f(tr.isp_scale ?? 1, 3), "×"),
    row("Solved", solved ? solved.replace("T", " ") : "—"),
  ].join("");

  const outcome = [
    row("Result", `<span class="badge ${cls}">${label}</span>`),
    row("Touchdown time", f(m.touchdown_time), "s"),
    row("Miss distance", f(m.touchdown_pos_err_m, 3), "m"),
    row("Vertical speed", f(m.touchdown_vert_speed, 3), "m/s"),
    row("Horizontal speed", f(m.touchdown_horiz_speed, 3), "m/s"),
    row("Total speed", f(vtd, 3), "m/s"),
    row("Tilt", f(m.touchdown_tilt_deg, 2), "°"),
    row("Body rate", f(m.touchdown_body_rate_deg_s, 2), "°/s"),
    row("Fuel used", f(m.fuel_used_kg, 0), "kg"),
    row("Fuel margin", f(m.fuel_margin_kg, 0), "kg"),
  ].join("");

  const plan = p ? [
    row("Ignition", `T+${f(pi.t_ignition)}`, "s"),
    row("Latest feasible", `T+${f(pi.t_latest_ignition)}`, "s"),
    row("Ignition altitude", f(p.r[0][0], 0), "m"),
    row("Burn time (t_f)", f(p.t.at(-1)), "s"),
    row("Planned fuel", f(fuelPlan, 0), "kg"),
    row("Nodes", pi.N ?? "—"),
    m.plan === "gfold" ? row("t_f search", `${pi.n_solves ?? "—"} SOCPs`) : row("SCvx iterations", `${pi.iterations ?? "—"} · ${pi.converged ? "converged" : "not converged"}`),
    row("Ignition search", `${pi.ignition_solves ?? "—"} solves`),
    row("Lossless gap ‖u‖−σ", pi.tightness_max != null ? pi.tightness_max.toExponential(1) : "—"),
    row("Open-loop end error", pi.open_loop_terminal_err ? `${f(pi.open_loop_terminal_err[0], 3)} m · ${f(pi.open_loop_terminal_err[1], 3)} m/s` : "—"),
    row("Solve time", f(pi.solve_time_s, 1), "s"),
  ].join("") : `<p class="muted">No plan data.</p>`;

  const control = [
    row("Max gimbal", f(m.max_gimbal_deg, 2), `° / ${rocket.attitude?.gimbal_max_deg ?? 8}°`),
    row("Gimbal saturated", f(100 * (m.gimbal_sat_frac ?? 0), 1), "% of burn"),
    row("Max slew rate", f(m.max_slew_rate_deg_s, 1), "°/s"),
    row("Max tracking error", f(m.max_tracking_err_m, 2), "m"),
    row("Re-plans", m.n_replans ? `${m.n_replans} (${m.n_replan_failures} rejected, ${m.n_replans_relaxed} relaxed)` : "—"),
    m.controller === "mpc" ? row("MPC solves", `${m.mpc_solves} · ${m.mpc_failures} failed`) : "",
    m.controller === "mpc" ? row("MPC mean solve", f(m.mpc_mean_solve_ms, 1), "ms") : "",
    row("Sim wall time", f(m.wall_time_s, 1), "s"),
  ].join("");

  const all = Object.entries(m).filter(([, v]) => typeof v !== "object")
    .map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "number" ? (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : +v.toFixed(4)) : v}</td></tr>`).join("");

  const charts = [
    ["alt", "Altitude", "m"], ["spd", "Speed: vertical — / horizontal - -", "m/s"], ["thr", "Throttle", "% T_max"],
    ["att", "Attitude: pitch — / yaw - -", "°"], ["gim", "Gimbal: pitch — / yaw - -", "°"], ["mass", "Mass", "kg"],
  ];

  el.innerHTML = `
    <header class="dhead">
      <span class="swatch" style="background:${c.color}"></span>
      <div><h1>${sim.name}</h1><p class="muted">${describe(m)}</p></div>
      <span class="badge ${cls}">${label}</span>
    </header>
    <div class="dgrid">
      ${card("Configuration", config)}
      ${card("Touchdown", outcome)}
      ${card("Guidance plan", plan)}
      ${card("Control", control)}
    </div>
    <div class="dcharts">${charts.map(([id]) => `<canvas data-chart="${id}"></canvas>`).join("")}</div>
    ${card("All metrics", `<table class="mtable">${all}</table>`, "wide")}`;

  // time histories
  const t = sim.t, col = c.color, N = t.length;
  const py = sim.att ? sim.att.map(pitchYaw) : null;
  const gpy = sim.att ? t.map((_, k) => (sim.throttle[k] > 0.01 ? gimbalPY(gimbalAt(sim, k), sim.att[k]) : [0, 0])) : null;
  const tMax = t[N - 1] + 0.5;
  const series = {
    alt: [{ t, y: sim.r.map((r) => r[0]), color: col, width: 1.8 }],
    spd: [{ t, y: sim.v.map((v) => -v[0]), color: col, width: 1.8 }, { t, y: sim.v.map((v) => Math.hypot(v[1], v[2])), color: col, width: 1.5, dashed: true }],
    thr: [{ t, y: sim.throttle.map((x) => 100 * x), color: col, width: 1.6 }],
    att: py ? [{ t, y: py.map((a) => a[0]), color: col, width: 1.6 }, { t, y: py.map((a) => a[1]), color: col, width: 1.4, dashed: true }] : [],
    gim: gpy ? [{ t, y: gpy.map((a) => a[0]), color: col, width: 1.4 }, { t, y: gpy.map((a) => a[1]), color: col, width: 1.2, dashed: true }] : [],
    mass: [{ t, y: sim.m, color: col, width: 1.8 }],
  };
  const gmax = rocket.attitude?.gimbal_max_deg ?? 8;
  const opts = {
    alt: { yMin: 0 }, spd: { yMin: 0 }, thr: { yMin: 0, yMax: 105, bands: [{ y: rocket.thr_min * 100, label: "min" }] },
    att: {}, gim: { yMin: -gmax * 1.1, yMax: gmax * 1.1, bands: [{ y: gmax, label: `+${gmax}°` }, { y: -gmax, label: `−${gmax}°` }] }, mass: {},
  };
  const ign = m.t_ignition;
  for (const [id, title, unit] of charts) {
    const ch = new TimeChart(el.querySelector(`[data-chart="${id}"]`), { title, unit, cursor: null, ...opts[id] });
    ch.markers = ign > 0 ? [{ t: ign, label: "ignition" }] : [];
    ch.setSeries(series[id], tMax);
  }
}
