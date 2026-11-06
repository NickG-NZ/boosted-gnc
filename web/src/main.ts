import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { type Series, TimeChart } from "./charts.js";
import { type DetailsCase, renderDetails, type Verdict } from "./details.js";
import { GimbalTarget, GimbalView } from "./gimbal.js";
import { Rocket3D } from "./rocket3d.js";
import { createWorld, canvas2d } from "./scene.js";
import type { Metrics, Replan, Results, Sim, SolverStatus, Vec3, Wind } from "./types.js";
import { CASE_COLORS, gimbalPY, norm, pitchYaw, sampleSim, type SimSample, toThree } from "./util.js";

/** Element by id; throws if the page markup is missing it. */
function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
}

async function loadData(): Promise<Results> {
  const url = new URLSearchParams(location.search).get("data") || "data/results.json";
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status}). Run: uv run python run.py`);
  return res.json() as Promise<Results>;
}

interface FatLineOptions {
  width?: number;
  dashed?: boolean;
  dash?: number;
  gap?: number;
  opacity?: number;
}

function fatLine(points: THREE.Vector3[], color: string, { width = 2, dashed = false, dash = 18, gap = 12, opacity = 1 }: FatLineOptions = {}): Line2 {
  const geo = new LineGeometry();
  geo.setPositions(points.flatMap((p) => [p.x, p.y, p.z]));
  const mat = new LineMaterial({
    color, linewidth: width, dashed, dashSize: dash, gapSize: gap, transparent: true, opacity,
    resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
  });
  const line = new Line2(geo, mat);
  if (dashed) line.computeLineDistances();
  return line;
}

/** Smooth a coarse node path (guidance nodes) for display. */
const smoothPath = (nodes: Vec3[], n = 200) => new THREE.CatmullRomCurve3(nodes.map(toThree)).getPoints(n);

/** Numbered disc sprite used to mark other cases' rockets when comparing. */
function labelSprite(text: string, color: string): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 64;
  const g = canvas2d(c);
  g.fillStyle = color;
  g.beginPath(); g.arc(32, 32, 28, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#06080c";
  g.font = "bold 34px Inter, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 32, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation: false }));
  s.renderOrder = 10;
  return s;
}

function verdict(m: Metrics): Verdict {
  if (!m.landed) return ["bad", "no touchdown"];
  const v = Math.hypot(m.touchdown_vert_speed, m.touchdown_horiz_speed);
  if (v < 2 && m.touchdown_pos_err_m < 2) return ["good", `landed ${v.toFixed(2)} m/s`];
  if (v < 5) return ["warn", `hard ${v.toFixed(2)} m/s`];
  return ["bad", `crash ${v.toFixed(0)} m/s`];
}

const PLAN_NAMES: Record<string, string> = { gfold: "G-FOLD (no drag model)", scvx: "SCvx (drag model)" };

const CTRL_NAMES: Record<string, string> = { mpc: "MPC", cascade: "Cascade" };

/** Split "Scenario · details · controller" into a group label and the controller label. */
function splitName(sim: Sim): { base: string; ctrl: string } {
  const ctrl = sim.metrics.controller;
  let base = sim.name;
  const parts = base.split(" · ");
  if (ctrl && parts.length > 1 && parts[parts.length - 1].toLowerCase() === ctrl.toLowerCase()) base = parts.slice(0, -1).join(" · ");
  return { base, ctrl: (ctrl && CTRL_NAMES[ctrl]) ?? ctrl ?? "run" };
}

interface Tag {
  t: string;
  k?: string;
}

/** Tags for a run: controller, plus re-planning when on. Guidance/environment are in the title. */
function runTags(m: Metrics, ctrl: string): Tag[] {
  const tags: Tag[] = [{ t: ctrl, k: "ctrl" }];
  if (m.replan_hz) tags.push({ t: `Re-plan ${m.replan_hz} Hz` });
  return tags;
}
const tagHtml = (tags: Tag[]) => tags.map((g) => `<span class="tag${g.k ? ` ${g.k}` : ""}">${g.t}</span>`).join("");

/** Short text for a case's truth wind: constant [x,y,z] or the shear+gust model (sim/crates/sim-env/src/wind.rs). */
function windSummary(w: Wind | undefined): string | null {
  if (!w) return null;
  if (Array.isArray(w)) {
    const v = norm(w);
    return v ? `wind ${v.toFixed(0)} m/s` : null;
  }
  const v = norm(w.mean ?? [0, 0, 0]);
  const parts = [`wind ${v.toFixed(0)} m/s @ ${w.ref_alt ?? 10} m`];
  if (w.shear) parts.push(`shear ^${w.shear}`);
  if (w.gust_sigma) parts.push(`gusts σ ${w.gust_sigma} m/s`);
  return parts.join(", ");
}

function describe(m: Metrics): string {
  const tr = m.truth || {};
  let truth = tr.drag === false ? "vacuum" : "atmosphere";
  const pert: string[] = [];
  if (tr.cd_scale && tr.cd_scale !== 1) pert.push(`C_D ×${tr.cd_scale}`);
  const wind = windSummary(tr.wind);
  if (wind) pert.push(wind);
  if (tr.isp_scale && tr.isp_scale !== 1) pert.push(`Isp ×${tr.isp_scale}`);
  if (pert.length) truth += ` (${pert.join(", ")})`;
  const loop = m.replan_hz ? `re-plans at ${m.replan_hz} Hz` : "plan fixed in advance";
  return `Plan: ${PLAN_NAMES[m.plan] ?? m.plan} · Truth: ${truth} · ${loop}`;
}

interface ReplanLine extends Replan {
  line: Line2;
}

/** One simulation run with its scene objects and panel row. */
interface Case extends DetailsCase {
  flown: Line2;
  planGroup: THREE.Group;
  replanGroup: THREE.Group;
  replanLines: ReplanLine[];
  ignition: THREE.Mesh | null;
  marker: THREE.Sprite;
  tEnd: number;
  base: string;
  index: number;
  tags: Tag[];
  rowHtml: string;
  el: HTMLButtonElement;
}

type Tab = "view" | "details";

async function main(): Promise<void> {
  const data = await loadData();
  const { scenario, rocket, plans, sims } = data;
  const h0 = scenario.r0[0];
  const world = createWorld($("viewport"), { glideSlopeDeg: scenario.glide_slope_deg, coneHeight: Math.max(1800, h0 * 1.1) });
  const { scene, camera, controls, renderer } = world;
  const lineMats: LineMaterial[] = [];
  const track = (...ls: Line2[]) => ls.forEach((l) => lineMats.push(l.material));

  // ---------------------------------------------------------------- per-case scene objects
  const cases: Case[] = sims.map((sim, i) => {
    const color = CASE_COLORS[i % CASE_COLORS.length];
    const m = sim.metrics;
    const planData = sim.plan ?? plans[m.plan]; // each run carries the plan it flew
    const flown = fatLine(sim.r.map(toThree), color, { width: 2.5 });
    const planGroup = new THREE.Group();
    // predicted coast (guidance's own model) dotted, then the burn plan dashed
    const coastPath = planData?.info.coast;
    if (coastPath && coastPath.r.length > 1) {
      const coast = fatLine(coastPath.r.map(toThree), color, { width: 1.5, dashed: true, dash: 4, gap: 14, opacity: 0.6 });
      planGroup.add(coast); track(coast);
    }
    if (planData) {
      const burn = fatLine(smoothPath(planData.r), color, { width: 1.5, dashed: true, opacity: 0.6 });
      planGroup.add(burn); track(burn);
    }

    const replanGroup = new THREE.Group();
    const replanLines = (sim.replans || []).map((rp) => {
      const l = fatLine(smoothPath(rp.r, 80), color, { width: 1.4, opacity: 0.15 });
      replanGroup.add(l); track(l);
      return { ...rp, line: l };
    });

    // ignition marker: ring at the true position where the engine lit
    let ignition: THREE.Mesh | null = null;
    if (m.t_ignition > 0) {
      const k = sim.t.findIndex((tt) => tt >= m.t_ignition - 1e-6);
      ignition = new THREE.Mesh(new THREE.TorusGeometry(14, 2.2, 8, 32), new THREE.MeshBasicMaterial({ color: "#ff9a4d" }));
      ignition.position.copy(toThree(sim.r[Math.max(k, 0)]));
      ignition.rotation.x = Math.PI / 2;
      scene.add(ignition);
    }

    const marker = labelSprite(String(i + 1), color);
    marker.scale.setScalar(0.035);
    track(flown);
    scene.add(flown, planGroup, replanGroup, marker);
    // panel fields (base, ctrl, el, ...) are filled in when the run rows are built below
    return {
      sim, color, planData, flown, planGroup, replanGroup, replanLines, ignition, marker, tEnd: sim.t[sim.t.length - 1],
    } as Case;
  });

  const rocket3d = new Rocket3D(rocket.geometry);
  const gimbalMax = rocket.attitude?.gimbal_max_deg ?? rocket.gimbal_max_deg ?? 8;
  const gimbalView = new GimbalView($<HTMLCanvasElement>("gimbal-3d"), gimbalMax);
  const gimbalTarget = new GimbalTarget($<HTMLCanvasElement>("gimbal-target"), gimbalMax);
  $("g-max").textContent = `/ ${gimbalMax}°`;
  // exaggerated flame level: 0 at the engine floor, 1 at full thrust (throttle spans ~0.4-0.9)
  const thrLo = (rocket.thr_min ?? 0.4) * 0.9;
  const flameLevel = (thr: number) => Math.min(1, Math.max(0, (thr - thrLo) / (1 - thrLo)));
  scene.add(rocket3d.group);

  const tMax = Math.max(...cases.map((c) => c.tEnd)) + 0.5;
  let selected = Math.max(0, cases.findIndex((c) => c.sim.metrics.plan === "scvx"));
  let t = 0, playing = true, speed = 1, chase = false;
  let tab: Tab = "view", detailsFor = -1;

  // ---------------------------------------------------------------- tabs: 3D view | details
  function showDetails() {
    if (detailsFor === selected) return;
    detailsFor = selected;
    renderDetails($("details"), cases[selected], { rocket, scenario, verdict, describe, windSummary });
  }
  function setTab(name: Tab) {
    tab = name;
    document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
    const det = name === "details";
    $("details").hidden = !det;
    $("right").hidden = det;
    $("timeline").hidden = det;
    if (det) { setPlaying(false); showDetails(); }
    try { localStorage.setItem("bgnc-tab", name); } catch { /* storage unavailable */ }
  }
  document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab as Tab)));
  const lastRocketPos = new THREE.Vector3();

  // ---------------------------------------------------------------- panel
  // one row per run: title + descriptive tags (results live in the Details tab)
  cases.forEach((c, i) => {
    let { base, ctrl } = splitName(c.sim);
    base = base.replace(/ · re-plan [\d.]+ Hz$/i, "");
    Object.assign(c, { base, ctrl, index: i, tags: runTags(c.sim.metrics, ctrl) });
    const el = document.createElement("button");
    el.className = "run";
    el.title = c.sim.name;
    c.rowHtml = `<span class="run-body"><span class="run-title">${base}</span><span class="tags">${tagHtml(c.tags)}<span class="run-status"></span></span></span>
      <span class="run-num">${i + 1}</span>`;
    el.innerHTML = c.rowHtml;
    el.addEventListener("click", () => select(i));
    $("cases").appendChild(el);
    c.el = el;
  });

  // ---------------------------------------------------------------- charts
  const altChart = new TimeChart($<HTMLCanvasElement>("chart-alt"), { title: "Altitude", unit: "m", yMin: 0 });
  const thrChart = new TimeChart($<HTMLCanvasElement>("chart-thr"), {
    title: "Throttle", unit: "% T_max", yMin: 0, yMax: 105,
    bands: [{ y: rocket.thr_min * 100, label: "min" }, { y: rocket.thr_max * 100, label: "plan max" }],
  });

  const checkbox = (id: string) => $<HTMLInputElement>(id);
  const comparing = () => checkbox("opt-compare").checked;

  function refreshCharts() {
    const shown = cases.map((c, i) => ({ c, i })).filter(({ i }) => comparing() || i === selected);
    shown.sort((a, b) => Number(a.i === selected) - Number(b.i === selected)); // selected drawn last (on top)
    const mk = (f: "alt" | "thr"): Series[] => shown.map(({ c, i }) => ({
      t: c.sim.t, y: f === "alt" ? c.sim.r.map((v) => v[0]) : c.sim.throttle.map((v) => v * 100),
      color: c.color, width: i === selected ? 2 : 1, alpha: i === selected ? 1 : 0.35,
    }));
    const ign = cases[selected].sim.metrics.t_ignition;
    altChart.markers = thrChart.markers = ign > 0 ? [{ t: ign, label: "ignition" }] : [];
    altChart.setSeries(mk("alt"), tMax);
    thrChart.setSeries(mk("thr"), tMax);
  }

  // ---------------------------------------------------------------- selection / visibility
  function select(i: number) {
    selected = i;
    cases.forEach((c, j) => (c.el.hidden = j === i));
    const active = $("active-case");
    active.innerHTML = cases[i].rowHtml;
    active.title = cases[i].sim.name;
    delete active.dataset.solve; // solve status is re-applied on the next solver poll
    if (tab === "details") showDetails();
    refresh();
  }

  function refresh() {
    const cmp = comparing();
    world.cone.visible = checkbox("opt-cone").checked;
    cases.forEach((c, i) => {
      const sel = i === selected;
      c.flown.visible = sel || cmp;
      c.flown.material.opacity = sel ? 1 : 0.35;
      c.planGroup.visible = sel && checkbox("opt-plans").checked;
      c.replanGroup.visible = sel && checkbox("opt-replans").checked;
      if (c.ignition) c.ignition.visible = sel;
    });
    refreshCharts();
  }

  ["opt-plans", "opt-replans", "opt-cone", "opt-compare"].forEach((id) => $(id).addEventListener("change", refresh));
  checkbox("opt-chase").addEventListener("change", (e) => setChase((e.target as HTMLInputElement).checked));
  const scaleInput = $<HTMLInputElement>("opt-scale");
  rocket3d.setScale(+scaleInput.value);

  const overview = () => {
    controls.target.set(60, h0 * 0.45, -30);
    camera.position.set(h0 * 1.25, h0 * 0.4, h0 * 1.4);
  };
  overview();

  function setChase(on: boolean) {
    chase = on;
    checkbox("opt-chase").checked = on;
    if (on) {
      const p = rocket3d.group.position;
      const bodyLen = (rocket.geometry?.length ?? 40) * +scaleInput.value;
      controls.target.copy(p).add(new THREE.Vector3(0, bodyLen * 0.5, 0));
      camera.position.copy(controls.target).add(new THREE.Vector3(1.9, 0.35, 2.2).multiplyScalar(bodyLen));
    } else overview();
    lastRocketPos.copy(rocket3d.group.position);
  }

  // ---------------------------------------------------------------- timeline
  const scrub = $<HTMLInputElement>("scrub");
  const btnPlay = $<HTMLButtonElement>("btn-play");
  const setPlaying = (p: boolean) => { playing = p; btnPlay.textContent = p ? "❚❚" : "▶"; };
  btnPlay.onclick = () => { if (t >= tMax) t = 0; setPlaying(!playing); };
  $("btn-restart").onclick = () => { t = 0; setPlaying(true); };
  $<HTMLSelectElement>("speed").onchange = (e) => (speed = +(e.target as HTMLSelectElement).value);
  scrub.addEventListener("input", () => { t = (+scrub.value / 1000) * tMax; setPlaying(false); });
  window.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement | null)?.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); btnPlay.click(); }
    else if (e.key === "ArrowRight") { t = Math.min(t + 0.1, tMax); setPlaying(false); }
    else if (e.key === "ArrowLeft") { t = Math.max(t - 0.1, 0); setPlaying(false); }
    else if (e.key.toLowerCase() === "c") setChase(!chase);
    else if (e.key === "ArrowDown") { e.preventDefault(); select((selected + 1) % cases.length); }
    else if (e.key === "ArrowUp") { e.preventDefault(); select((selected + cases.length - 1) % cases.length); }
    else if (e.key.toLowerCase() === "d") setTab(tab === "details" ? "view" : "details");
    else if (/^[1-9]$/.test(e.key) && +e.key <= cases.length) select(+e.key - 1);
  });
  window.addEventListener("resize", () => lineMats.forEach((m) => m.resolution.set(window.innerWidth, window.innerHeight)));

  // ---------------------------------------------------------------- HUD
  function updateHud(c: Case, st: SimSample) {
    const m = c.sim.metrics;
    const nT = norm(st.T);
    const ax: Vec3 = st.att ?? (nT > 1 ? [st.T[0] / nT, st.T[1] / nT, st.T[2] / nT] : [1, 0, 0]);
    const [pitch, yaw] = pitchYaw(ax);
    $("hud-t").textContent = Math.min(t, c.tEnd).toFixed(2);
    $("hud-alt").textContent = st.r[0].toFixed(1);
    $("hud-vz").textContent = st.v[0].toFixed(1);
    $("hud-vh").textContent = Math.hypot(st.v[1], st.v[2]).toFixed(1);
    $("hud-thr").textContent = (st.throttle * 100).toFixed(0);
    $("hud-pitch").textContent = pitch.toFixed(1);
    $("hud-yaw").textContent = yaw.toFixed(1);
    $("hud-fuel").textContent = (st.m - rocket.m_dry).toFixed(0);

    // engine gimbal close-up
    const [gp, gy] = gimbalPY(st.gimbal, ax);
    const gmag = Math.asin(Math.min(1, norm(st.gimbal))) * 180 / Math.PI;
    const sat = gmag >= 0.98 * gimbalMax;
    $("g-p").textContent = gp.toFixed(1);
    $("g-y").textContent = gy.toFixed(1);
    $("g-mag").textContent = gmag.toFixed(1);
    $("g-mag").classList.toggle("sat", sat);
    gimbalView.update(gp, gy, flameLevel(st.throttle), st.throttle > 0.01);
    const trail: [number, number][] = [];
    for (let k = 24; k >= 0; k--) {
      const tt = t - k * 0.06;
      if (tt < 0) continue;
      const s2 = sampleSim(c.sim, Math.min(tt, c.tEnd));
      if (s2.throttle > 0.01 && s2.att) trail.push(gimbalPY(s2.gimbal, s2.att));
    }
    gimbalTarget.draw(gp, gy, st.throttle > 0.01 ? trail : [], c.color, sat);

    const phase = $("hud-phase"), status = $("hud-status");
    status.style.color = "var(--muted)";
    if (st.done) {
      const [cls, label] = verdict(m);
      phase.className = `phase ${cls}`;
      phase.textContent = m.landed ? "TOUCHDOWN" : "NO TOUCHDOWN";
      status.style.color = `var(--${cls})`;
      status.textContent = `${label} · miss ${m.touchdown_pos_err_m.toFixed(2)} m`;
    } else if (t < m.t_ignition) {
      phase.className = "phase coast";
      phase.textContent = "COAST · ENGINE OFF";
      status.textContent = `ignition in ${(m.t_ignition - t).toFixed(1)} s`;
    } else if (st.throttle > 0) {
      phase.className = "phase burn";
      phase.textContent = "POWERED DESCENT";
      const started = c.replanLines.filter((r) => r.t0 <= t);
      const active = started[started.length - 1];
      status.textContent = active ? `following plan made at T+${active.t0.toFixed(1)} s` : "following pre-computed plan";
    } else {
      phase.className = "phase coast";
      phase.textContent = "ENGINE CUT · FALLING";
      status.textContent = "";
    }
  }

  // ---------------------------------------------------------------- loop
  const clock = new THREE.Clock();
  const q = new URLSearchParams(location.search); // deep links: ?case=3&t=10.5&chase=1&paused=1&compare=1
  if (q.has("case")) selected = Math.min(Math.max(+(q.get("case") ?? 1) - 1, 0), cases.length - 1);
  if (q.has("t")) t = +(q.get("t") ?? 0);
  if (q.get("compare") === "1") checkbox("opt-compare").checked = true;
  select(selected);
  setPlaying(!q.has("paused"));
  let savedTab = q.get("tab");
  try { savedTab ??= localStorage.getItem("bgnc-tab"); } catch { /* storage unavailable */ }
  if (savedTab === "details") setTab("details");
  void initSolver({
    runs: cases.map((c) => ({ name: c.sim.name, el: c.el })), selectedName: () => cases[selected].sim.name,
    selectedIndex: () => selected, time: () => t, tab: () => tab, solvedAt: data.solved_at,
  });
  if (q.get("chase") !== "0") { // chase camera by default; ?chase=0 or key C for the overview
    const st0 = sampleSim(cases[selected].sim, t);
    rocket3d.update(toThree(st0.r), st0.att ? toThree(st0.att) : new THREE.Vector3(0, 1, 0), 0);
    setChase(true);
  }

  renderer.setAnimationLoop(() => {
    const dt = clock.getDelta();
    if (tab === "details") return; // 3D view hidden: skip rendering
    if (playing) {
      t += dt * speed;
      if (t >= tMax) { t = tMax; setPlaying(false); }
    }
    scrub.value = String(Math.round((t / tMax) * 1000));
    $("time-label").textContent = `${t.toFixed(2)} s`;

    const cmp = comparing();
    cases.forEach((c, i) => {
      const st = sampleSim(c.sim, t);
      const pos = toThree(st.r);
      c.marker.visible = cmp && i !== selected;
      c.marker.position.copy(pos);
      c.replanLines.forEach((r, k) => {
        const active = r.t0 <= t && (k === c.replanLines.length - 1 || c.replanLines[k + 1].t0 > t);
        r.line.material.opacity = active ? 0.95 : 0.12;
      });
      if (i === selected) {
        // body axis from the sim's attitude log (fallback: thrust direction for old data)
        const axis = st.att ? toThree(st.att) : (norm(st.T) > 1 ? toThree(st.T) : null);
        rocket3d.update(pos, axis, st.throttle, flameLevel(st.throttle), toThree(st.gimbal));
        updateHud(c, st);
        const s = world.shadow;
        s.position.set(pos.x, 0.2, pos.z);
        s.scale.setScalar(4 * +scaleInput.value + pos.y * 0.02);
        s.material.opacity = Math.max(0.05, 0.55 - pos.y / 1500);
      }
    });

    if (chase) {
      const p = rocket3d.group.position;
      const d = p.clone().sub(lastRocketPos);
      camera.position.add(d);
      controls.target.add(d);
      lastRocketPos.copy(p);
    }
    altChart.setCursor(t);
    thrChart.setCursor(t);
    controls.update();
    renderer.render(scene, camera);
  });
}

// ---------------------------------------------------------------- solver (needs `run.py --serve`)
const api = {
  status: (): Promise<SolverStatus | null> =>
    fetch("/api/status", { cache: "no-store" }).then((r) => (r.ok ? (r.json() as Promise<SolverStatus>) : null)).catch(() => null),
  solve: (cases: string[] | null): Promise<SolverStatus & { error?: string | null }> =>
    fetch("/api/solve", { method: "POST", body: JSON.stringify({ cases }) }).then((r) => r.json()),
  cancel: (): Promise<SolverStatus> => fetch("/api/cancel", { method: "POST", body: "{}" }).then((r) => r.json()),
};

const fmtAgo = (iso: string | null | undefined): string => {
  if (!iso) return "unknown time";
  const d = new Date(iso), s = (Date.now() - d.getTime()) / 1000;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};

interface SolverContext {
  runs: { name: string; el: HTMLElement }[];
  selectedName: () => string;
  selectedIndex: () => number;
  time: () => number;
  tab: () => string;
  solvedAt: string | undefined;
}

type RowState = "done" | "active" | "queued";

/** Wires the Run selected / Run all buttons. When a solve finishes the page reloads on the same
 * run, tab and time so the new results show. */
async function initSolver(ctx: SolverContext): Promise<void> {
  const st0 = await api.status();
  const btnSel = $<HTMLButtonElement>("btn-solve-sel"), btnAll = $<HTMLButtonElement>("btn-solve-all");
  const btnCancel = $<HTMLButtonElement>("btn-cancel");
  const status = $("solver-status");
  const idle = () => (status.textContent = `Results from ${fmtAgo(ctx.solvedAt ?? st0?.data_mtime)}`);
  if (!st0) {
    btnSel.disabled = btnAll.disabled = true;
    status.innerHTML = `Static mode: start with <code>run.py --serve</code> to solve`;
    return;
  }
  const version = st0.version;
  let timer: ReturnType<typeof setTimeout> | null = null;

  /** Mark each run row: queued / active (being planned or simulated) / done, or clear it. */
  function markRows(st: SolverStatus) {
    const queue = st.running ? (st.cases ?? ctx.runs.map((r) => r.name)) : [];
    const p = st.progress;
    const cur = p?.stage === "sim" ? queue.indexOf(p.label) : p?.stage === "done" ? queue.length : -1; // -1: still planning
    for (const r of ctx.runs) {
      const k = queue.indexOf(r.name);
      let state: RowState | null = null, text = "";
      if (k >= 0 && cur >= 0 && k < cur) [state, text] = ["done", "Updated"];
      else if (k >= 0 && k === cur) [state, text] = ["active", "Simulating…"];
      else if (k >= 0 && cur < 0 && p?.stage === "plan") [state, text] = ["active", `${p.label}…`];
      else if (k >= 0) [state, text] = ["queued", p ? "Queued" : "Starting…"];
      for (const el of [r.el, r.name === ctx.selectedName() ? $("active-case") : null]) {
        if (!el) continue; // the selected run is also shown in the "Active simulation" card
        if (state) el.dataset.solve = state; else delete el.dataset.solve;
        const statusEl = el.querySelector(".run-status");
        if (statusEl) statusEl.textContent = text;
      }
    }
  }

  function show(st: SolverStatus) {
    const running = st.running;
    btnSel.disabled = btnAll.disabled = running;
    btnSel.hidden = btnAll.hidden = running; // header shows only Cancel while solving
    btnCancel.hidden = !running;
    markRows(st);
    if (running) {
      const p = st.progress;
      status.textContent = p ? `Solving ${p.done + 1}/${p.total}` : "Starting solver…";
      status.classList.remove("err");
    } else if (st.error) {
      status.textContent = st.error === "cancelled" ? "Solve cancelled" : `Solve failed: ${st.error}`;
      status.classList.toggle("err", st.error !== "cancelled");
    } else idle();
  }

  async function poll() {
    const st = await api.status();
    if (!st) return;
    show(st);
    if (st.running) { timer = setTimeout(poll, 400); return; }
    timer = null;
    if (st.version !== version && !st.error) {
      const q = new URLSearchParams(location.search);
      q.set("case", String(ctx.selectedIndex() + 1));
      q.set("t", ctx.time().toFixed(2));
      q.set("tab", ctx.tab());
      q.set("paused", "1");
      location.search = q.toString();
    }
  }
  async function start(cases: string[] | null) {
    const r = await api.solve(cases);
    if (r.error) { status.textContent = r.error; return; }
    show(r);
    if (!timer) void poll();
  }
  btnSel.onclick = () => start([ctx.selectedName()]);
  btnAll.onclick = () => start(null);
  btnCancel.onclick = () => api.cancel();
  show(st0);
  if (st0.running) void poll();
  setInterval(() => !timer && !btnSel.disabled && idle(), 30000);
}

main().catch(async (err: unknown) => {
  console.error(err);
  const el = $("error");
  el.hidden = false;
  el.textContent = err instanceof Error ? err.message : String(err);
  if (await api.status()) { // served by run.py but no results yet: offer to solve
    const b = document.createElement("button");
    b.className = "btn primary";
    b.textContent = "Run all";
    b.onclick = async () => {
      b.disabled = true;
      await api.solve(null);
      const wait = async () => {
        const st = await api.status();
        if (st?.running) {
          b.textContent = st.progress ? `Solving ${st.progress.done + 1}/${st.progress.total}…` : "Starting…";
          setTimeout(wait, 500);
        } else location.reload();
      };
      void wait();
    };
    el.append(document.createElement("br"), b);
  }
});
