import * as THREE from "three";
import type { Sim, Vec3 } from "./types.js";

/** Guidance frame is [up, x, y]; three.js is Y-up. Map (up, x, y) -> (x, up, y). */
export const toThree = (p: Vec3): THREE.Vector3 => new THREE.Vector3(p[1], p[0], p[2]);

export const CASE_COLORS = ["#4cc9f0", "#f72585", "#ffb703", "#80ed99", "#b388ff", "#ff7b39", "#ffd6e8", "#9ef01a", "#48bfe3"];

/** Index k such that t[k] <= x < t[k+1] (clamped), via binary search. */
export function bracket(t: number[], x: number): number {
  if (x <= t[0]) return 0;
  if (x >= t[t.length - 1]) return t.length - 2;
  let lo = 0, hi = t.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= x) lo = mid; else hi = mid;
  }
  return lo;
}

/** State of a simulation record at one instant (guidance frame). */
export interface SimSample {
  att: Vec3 | null;
  gimbal: Vec3;
  r: Vec3;
  v: Vec3;
  m: number;
  T: Vec3;
  throttle: number;
  done: boolean;
}

const map3 = (v: Vec3, f: (c: number, i: number) => number): Vec3 => [f(v[0], 0), f(v[1], 1), f(v[2], 2)];

/** Sample a simulation record at time x. Positions/velocities/mass are linearly interpolated;
 * thrust is zero-order held (it is constant over each integration step). */
export function sampleSim(sim: Sim, x: number): SimSample {
  const t = sim.t;
  const k = bracket(t, x);
  const s = Math.min(Math.max((x - t[k]) / (t[k + 1] - t[k] || 1), 0), 1);
  const lerp3 = (a: Vec3[]): Vec3 => map3(a[k], (c, i) => c + s * (a[k + 1][i] - c));
  const done = x >= t[t.length - 1];
  // body axis (logged by the sim, slew-rate limited): normalized lerp between samples
  let att: Vec3 | null = null;
  if (sim.att) {
    const a = done ? sim.att[t.length - 1] : lerp3(sim.att);
    const n = norm(a) || 1;
    att = map3(a, (c) => c / n);
  }
  return {
    att,
    gimbal: done ? [0, 0, 0] : gimbalAt(sim, k),
    r: done ? sim.r[t.length - 1] : lerp3(sim.r),
    v: done ? sim.v[t.length - 1] : lerp3(sim.v),
    m: done ? sim.m[t.length - 1] : sim.m[k] + s * (sim.m[k + 1] - sim.m[k]),
    T: done ? [0, 0, 0] : sim.T[k],
    throttle: done ? 0 : sim.throttle[k],
    done,
  };
}

export const norm = (v: readonly number[]): number => Math.hypot(...v);

const dot = (u: Vec3, v: Vec3): number => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

/** Engine gimbal vector at log index k (guidance frame, perpendicular to the body axis,
 * |g| = sin(deflection)). Uses a logged `gimbal` array when present; otherwise it is recovered
 * exactly from the logged thrust direction t = sqrt(1-|g|^2) b + g and body axis b (both logged
 * after the same step): g = t - (t.b) b. */
export function gimbalAt(sim: Sim, k: number): Vec3 {
  if (sim.gimbal) return sim.gimbal[k];
  const T = sim.T[k], n = norm(T);
  if (!sim.att || n < 1) return [0, 0, 0];
  const b = sim.att[k], nb = norm(b) || 1;
  const th = map3(T, (c) => c / n), bu = map3(b, (c) => c / nb);
  const d = dot(th, bu);
  return map3(th, (c, i) => c - d * bu[i]);
}

const DEG = 180 / Math.PI;

/** Pitch / yaw of the body axis b = [up, x, y]: tilt toward +x and toward +y (deg). */
export const pitchYaw = (b: Vec3): [number, number] => [Math.atan2(b[1], b[0]) * DEG, Math.atan2(b[2], b[0]) * DEG];

/** Gimbal deflection split into pitch / yaw (deg): components of g along the pad x and y
 * directions projected into the plane perpendicular to the body axis. */
export function gimbalPY(g: Vec3, b: Vec3): [number, number] {
  const proj = (e: Vec3): Vec3 => {
    const d = dot(e, b);
    return map3(e, (c, i) => c - d * b[i]);
  };
  const unit = (v: Vec3): Vec3 => { const n = norm(v) || 1; return map3(v, (c) => c / n); };
  const ep = unit(proj([0, 1, 0]));
  let ey = proj([0, 0, 1]);
  const dp = dot(ey, ep);
  ey = unit(map3(ey, (c, i) => c - dp * ep[i]));
  const asinD = (x: number) => Math.asin(Math.max(-1, Math.min(1, x))) * DEG;
  return [asinD(dot(g, ep)), asinD(dot(g, ey))];
}
