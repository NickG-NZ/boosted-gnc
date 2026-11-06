import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { createWorld } from "./scene.js";
import { Rocket3D } from "./rocket3d.js";
import { TimeChart } from "./charts.js";
import { GimbalView, GimbalTarget } from "./gimbal.js";
import { CASE_COLORS, toThree, sampleSim, norm, gimbalPY, pitchYaw } from "./util.js";

const $ = (id) => document.getElementById(id);

async function loadData() {
  const url = new URLSearchParams(location.search).get("data") || "data/results.json";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status}). Run: uv run python run.py`);
  return res.json();
}

function fatLine(points, color, { width = 2, dashed = false, dash = 18, gap = 12, opacity = 1 } = {}) {
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
const smoothPath = (nodes, n = 200) => new THREE.CatmullRomCurve3(nodes.map(toThree)).getPoints(n);

/** Numbered disc sprite used to mark other cases' rockets when comparing. */
function labelSprite(text, color) {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 64;
  const g = c.getContext("2d");
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

function verdict(m) {
  if (!m.landed) return ["bad", "no touchdown"];
  const v = Math.hypot(m.touchdown_vert_speed, m.touchdown_horiz_speed);
  if (v < 2 && m.touchdown_pos_err_m < 2) return ["good", `landed ${v.toFixed(2)} m/s`];
  if (v < 5) return ["warn", `hard ${v.toFixed(2)} m/s`];
  return ["bad", `crash ${v.toFixed(0)} m/s`];
}

const PLAN_NAMES = { gfold: "G-FOLD (no drag model)", scvx: "SCvx (drag model)" };

const CTRL_NAMES = { mpc: "MPC", cascade: "Cascade" };

/** Split "Scenario · details · controller" into a group label and the controller label. */
function splitName(sim) {
  const ctrl = sim.metrics.controller;
  let base = sim.name;
  const parts = base.split(" · ");
  if (ctrl && parts.length > 1 && parts.at(-1).toLowerCase() === ctrl.toLowerCase()) base = parts.slice(0, -1).join(" · ");
  return { base, ctrl: CTRL_NAMES[ctrl] ?? ctrl ?? "run" };
}

function describe(m) {
  const tr = m.truth || {};
  let truth = tr.drag === false ? "vacuum" : "atmosphere";
  const pert = [];
  if (tr.cd_scale && tr.cd_scale !== 1) pert.push(`C_D ×${tr.cd_scale}`);
  if (tr.wind && tr.wind.some((w) => w)) pert.push(`wind ${Math.hypot(...tr.wind).toFixed(0)} m/s`);
  if (tr.isp_scale && tr.isp_scale !== 1) pert.push(`Isp ×${tr.isp_scale}`);
  if (pert.length) truth += ` (${pert.join(", ")})`;
  const loop = m.replan_hz ? `re-plans at ${m.replan_hz} Hz` : "plan fixed in advance";
  return `Plan: ${PLAN_NAMES[m.plan] ?? m.plan} · Truth: ${truth} · ${loop}`;
}

async function main() {
  const data = await loadData();
  const { scenario, rocket, plans, sims } = data;
  const h0 = scenario.r0[0];
  const world = createWorld($("viewport"), { glideSlopeDeg: scenario.glide_slope_deg, coneHeight: Math.max(1800, h0 * 1.1) });
  const { scene, camera, controls, renderer } = world;
  const lineMats = [];
  const track = (...ls) => ls.forEach((l) => lineMats.push(l.material));

  // ---------------------------------------------------------------- per-case scene objects
  const cases = sims.map((sim, i) => {
    const color = CASE_COLORS[i % CASE_COLORS.length];
    const m = sim.metrics;
    const planData = plans[m.plan];
    const flown = fatLine(sim.r.map(toThree), color, { width: 2.5 });
    const planGroup = new THREE.Group();
    // predicted coast (guidance's own model) dotted, then the burn plan dashed
    if (planData.info.coast && planData.info.coast.r.length > 1) {
      const coast = fatLine(planData.info.coast.r.map(toThree), color, { width: 1.5, dashed: true, dash: 4, gap: 14, opacity: 0.6 });
      planGroup.add(coast); track(coast);
    }
    const burn = fatLine(smoothPath(planData.r), color, { width: 1.5, dashed: true, opacity: 0.6 });
    planGroup.add(burn); track(burn);

    const replanGroup = new THREE.Group();
    const replanLines = (sim.replans || []).map((rp) => {
      const l = fatLine(smoothPath(rp.r, 80), color, { width: 1.4, opacity: 0.15 });
      replanGroup.add(l); track(l);
      return { ...rp, line: l };
    });

    // ignition marker: ring at the true position where the engine lit
    let ignition = null;
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
    return { sim, color, flown, planGroup, replanGroup, replanLines, ignition, marker, tEnd: sim.t.at(-1) };
  });

  const rocket3d = new Rocket3D(rocket.geometry);
  const gimbalMax = rocket.attitude?.gimbal_max_deg ?? rocket.gimbal_max_deg ?? 8;
  const gimbalView = new GimbalView($("gimbal-3d"), gimbalMax);
  const gimbalTarget = new GimbalTarget($("gimbal-target"), gimbalMax);
  $("g-max").textContent = `/ ${gimbalMax}°`;
  // exaggerated flame level: 0 at the engine floor, 1 at full thrust (throttle spans ~0.4-0.9)
  const thrLo = (rocket.thr_min ?? 0.4) * 0.9;
  const flameLevel = (thr) => Math.min(1, Math.max(0, (thr - thrLo) / (1 - thrLo)));
  scene.add(rocket3d.group);

  const tMax = Math.max(...cases.map((c) => c.tEnd)) + 0.5;
  let selected = Math.max(0, cases.findIndex((c) => c.sim.metrics.plan === "scvx"));
  let t = 0, playing = true, speed = 1, chase = false;
  const lastRocketPos = new THREE.Vector3();

  // ---------------------------------------------------------------- panel
  // runs grouped by scenario (name without the controller suffix); one segment per controller
  const groups = [];
  cases.forEach((c, i) => {
    const { base, ctrl } = splitName(c.sim);
    Object.assign(c, { base, ctrl, index: i });
    let g = groups.find((g) => g.base === base);
    if (!g) groups.push((g = { base, cases: [] }));
    g.cases.push(c);
  });
  groups.forEach((g) => {
    const el = document.createElement("div");
    el.className = "grp";
    const [head, ...rest] = g.base.split(" · ");
    el.innerHTML = `<div class="grp-name" title="${g.base}">${head}${rest.length ? ` <em>· ${rest.join(" · ")}</em>` : ""}</div><div class="seg"></div>`;
    g.cases.forEach((c) => {
      const m = c.sim.metrics;
      const [cls, label] = verdict(m);
      const b = document.createElement("button");
      b.style.setProperty("--c", c.color);
      b.title = `${c.index + 1}. ${c.sim.name} — ${label}, miss ${m.touchdown_pos_err_m.toFixed(2)} m`;
      b.innerHTML = `<span class="dot ${cls}"></span><span class="lbl">${c.ctrl}</span>
        <span class="val">${m.landed ? `${m.touchdown_pos_err_m.toFixed(m.touchdown_pos_err_m < 10 ? 2 : 0)} m` : "—"}</span>
        <span class="num">${c.index + 1}</span>`;
      b.addEventListener("click", () => select(c.index));
      el.querySelector(".seg").appendChild(b);
      c.btn = b;
      c.grpEl = el;
    });
    $("cases").appendChild(el);
  });

  function renderCurrent(c) {
    const m = c.sim.metrics;
    const [cls, label] = verdict(m);
    const [head, ...rest] = c.base.split(" · ");
    const vtd = Math.hypot(m.touchdown_vert_speed, m.touchdown_horiz_speed);
    $("current").innerHTML = `
      <div class="cur-eyebrow"><h2>Selected run · ${c.index + 1}/${cases.length}</h2><span class="badge ${cls}">${label}</span></div>
      <div class="cur-title">
        <span class="swatch" style="background:${c.color}"></span>
        <div class="cur-name">${head}${rest.length ? ` · ${rest.join(" · ")}` : ""}<small>${describe(m)}</small></div>
        <span class="ctrl-tag">${c.ctrl}</span>
      </div>
      <div class="cur-metrics">
        <div><span>Miss</span><b>${m.touchdown_pos_err_m.toFixed(2)}</b><i>m</i></div>
        <div><span>Touchdown</span><b>${vtd.toFixed(2)}</b><i>m/s</i></div>
        <div><span>Fuel</span><b>${m.fuel_used_kg.toFixed(0)}</b><i>kg</i></div>
        <div><span>Ignite</span><b>${m.t_ignition > 0 ? m.t_ignition.toFixed(1) : "—"}</b><i>T+ s</i></div>
      </div>`;
  }

  // ---------------------------------------------------------------- charts
  const altChart = new TimeChart($("chart-alt"), { title: "Altitude", unit: "m", yMin: 0 });
  const thrChart = new TimeChart($("chart-thr"), {
    title: "Throttle", unit: "% T_max", yMin: 0, yMax: 105,
    bands: [{ y: rocket.thr_min * 100, label: "min" }, { y: rocket.thr_max * 100, label: "plan max" }],
  });

  const comparing = () => $("opt-compare").checked;

  function refreshCharts() {
    const shown = cases.map((c, i) => ({ c, i })).filter(({ i }) => comparing() || i === selected);
    shown.sort((a, b) => (a.i === selected) - (b.i === selected)); // selected drawn last (on top)
    const mk = (f) => shown.map(({ c, i }) => ({
      t: c.sim.t, y: f === "alt" ? c.sim.r.map((v) => v[0]) : c.sim.throttle.map((v) => v * 100),
      color: c.color, width: i === selected ? 2 : 1, alpha: i === selected ? 1 : 0.35,
    }));
    const ign = cases[selected].sim.metrics.t_ignition;
    altChart.markers = thrChart.markers = ign > 0 ? [{ t: ign, label: "ignition" }] : [];
    altChart.setSeries(mk("alt"), tMax);
    thrChart.setSeries(mk("thr"), tMax);
  }

  // ---------------------------------------------------------------- selection / visibility
  function select(i) {
    selected = i;
    cases.forEach((c, j) => c.btn.classList.toggle("on", j === i));
    cases.forEach((c) => c.grpEl.classList.toggle("active", c.grpEl === cases[i].grpEl));
    renderCurrent(cases[i]);
    cases[i].grpEl.scrollIntoView({ block: "nearest" });
    refresh();
  }

  function refresh() {
    const cmp = comparing();
    world.cone.visible = $("opt-cone").checked;
    cases.forEach((c, i) => {
      const sel = i === selected;
      c.flown.visible = sel || cmp;
      c.flown.material.opacity = sel ? 1 : 0.35;
      c.planGroup.visible = sel && $("opt-plans").checked;
      c.replanGroup.visible = sel && $("opt-replans").checked;
      if (c.ignition) c.ignition.visible = sel;
    });
    refreshCharts();
  }

  ["opt-plans", "opt-replans", "opt-cone", "opt-compare"].forEach((id) => $(id).addEventListener("change", refresh));
  $("opt-chase").addEventListener("change", (e) => setChase(e.target.checked));
  const scaleInput = $("opt-scale");
  const applyScale = () => { rocket3d.setScale(+scaleInput.value); $("opt-scale-val").textContent = `${scaleInput.value}×`; };
  scaleInput.addEventListener("input", applyScale);
  applyScale();

  const overview = () => {
    controls.target.set(60, h0 * 0.45, -30);
    camera.position.set(h0 * 1.25, h0 * 0.4, h0 * 1.4);
  };
  overview();

  function setChase(on) {
    chase = on;
    $("opt-chase").checked = on;
    if (on) {
      const p = rocket3d.group.position;
      const bodyLen = (rocket.geometry?.length ?? 40) * +scaleInput.value;
      controls.target.copy(p).add(new THREE.Vector3(0, bodyLen * 0.5, 0));
      camera.position.copy(controls.target).add(new THREE.Vector3(1.9, 0.35, 2.2).multiplyScalar(bodyLen));
    } else overview();
    lastRocketPos.copy(rocket3d.group.position);
  }

  // ---------------------------------------------------------------- timeline
  const scrub = $("scrub");
  const setPlaying = (p) => { playing = p; $("btn-play").textContent = p ? "❚❚" : "▶"; };
  $("btn-play").onclick = () => { if (t >= tMax) t = 0; setPlaying(!playing); };
  $("btn-restart").onclick = () => { t = 0; setPlaying(true); };
  $("speed").onchange = (e) => (speed = +e.target.value);
  scrub.addEventListener("input", () => { t = (scrub.value / 1000) * tMax; setPlaying(false); });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); $("btn-play").click(); }
    else if (e.key === "ArrowRight") { t = Math.min(t + 0.1, tMax); setPlaying(false); }
    else if (e.key === "ArrowLeft") { t = Math.max(t - 0.1, 0); setPlaying(false); }
    else if (e.key.toLowerCase() === "c") setChase(!chase);
    else if (e.key === "ArrowDown") { e.preventDefault(); select((selected + 1) % cases.length); }
    else if (e.key === "ArrowUp") { e.preventDefault(); select((selected + cases.length - 1) % cases.length); }
    else if (/^[1-9]$/.test(e.key) && +e.key <= cases.length) select(+e.key - 1);
  });
  window.addEventListener("resize", () => lineMats.forEach((m) => m.resolution.set(window.innerWidth, window.innerHeight)));

  // ---------------------------------------------------------------- HUD
  function updateHud(c, st) {
    const m = c.sim.metrics;
    const ax = st.att ?? (norm(st.T) > 1 ? st.T.map((c) => c / norm(st.T)) : [1, 0, 0]);
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
    const trail = [];
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
      const active = c.replanLines.filter((r) => r.t0 <= t).at(-1);
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
  if (q.has("case")) selected = Math.min(Math.max(+q.get("case") - 1, 0), cases.length - 1);
  if (q.has("t")) t = +q.get("t");
  if (q.get("compare") === "1") $("opt-compare").checked = true;
  select(selected);
  setPlaying(!q.has("paused"));
  if (q.get("chase") !== "0") { // chase camera by default; ?chase=0 or key C for the overview
    const st0 = sampleSim(cases[selected].sim, t);
    rocket3d.update(toThree(st0.r), st0.att ? toThree(st0.att) : new THREE.Vector3(0, 1, 0), 0);
    setChase(true);
  }

  renderer.setAnimationLoop(() => {
    const dt = clock.getDelta();
    if (playing) {
      t += dt * speed;
      if (t >= tMax) { t = tMax; setPlaying(false); }
    }
    scrub.value = Math.round((t / tMax) * 1000);
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

main().catch((err) => {
  console.error(err);
  const el = $("error");
  el.hidden = false;
  el.textContent = String(err.message || err);
});
