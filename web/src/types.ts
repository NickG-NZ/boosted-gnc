/** Shapes of `data/results.json` (written by `run.py`) and of the `run.py --serve` API. */

/** A 3-vector in the guidance frame: [up, x, y]. */
export type Vec3 = [number, number, number];

export interface WindModel {
  mean?: Vec3;
  ref_alt?: number;
  shear?: number;
  gust_sigma?: number;
  gust_length?: number;
  seed?: number;
}

/** Truth wind: a constant vector or the shear + gust model. */
export type Wind = Vec3 | WindModel;

export interface Truth {
  drag?: boolean;
  cd_scale?: number;
  isp_scale?: number;
  wind?: Wind;
}

export interface Scenario {
  name: string;
  path?: string;
  m0: number;
  r0: Vec3;
  v0: Vec3;
  glide_slope_deg: number;
  wind?: Vec3;
}

export interface Geometry {
  length?: number;
  diameter?: number;
  n_legs?: number;
  n_grid_fins?: number;
}

export interface Attitude {
  gimbal_max_deg?: number;
  gimbal_rate_deg_s?: number;
  inertia?: number;
  arm?: number;
  body_rate_max_deg_s?: number;
  k_theta?: number;
  k_omega?: number;
  fin_alpha_max_deg_s2?: number;
  brake_frac?: number;
}

export interface Rocket {
  name: string;
  m_dry: number;
  T_max: number;
  thr_min: number;
  thr_max: number;
  theta_max_deg?: number;
  isp?: number;
  S_ref?: number;
  C_D?: number;
  gimbal_max_deg?: number;
  geometry?: Geometry;
  attitude?: Attitude;
}

export interface PlanInfo {
  solver_status?: string;
  N?: number;
  n_solves?: number;
  iterations?: number;
  converged?: boolean;
  tightness_max?: number;
  t_ignition?: number;
  t_latest_ignition?: number;
  ignition_solves?: number;
  coast?: { t: number[]; r: Vec3[] };
  solve_time_s?: number;
  open_loop_terminal_err?: [number, number];
  [key: string]: unknown;
}

export interface Plan {
  t: number[];
  r: Vec3[];
  v: Vec3[];
  m: number[];
  T: Vec3[];
  throttle: number[];
  hold?: "zoh" | "foh";
  info: PlanInfo;
}

export interface Replan {
  t0: number;
  tf: number;
  r: Vec3[];
  relaxed?: boolean;
}

export interface Metrics {
  landed: boolean;
  plan: string;
  controller?: string;
  touchdown_time: number;
  touchdown_pos_err_m: number;
  touchdown_vert_speed: number;
  touchdown_horiz_speed: number;
  touchdown_tilt_deg?: number;
  touchdown_body_rate_deg_s?: number;
  t_ignition: number;
  plan_tf?: number;
  max_slew_rate_deg_s?: number;
  max_gimbal_deg?: number;
  gimbal_sat_frac?: number;
  fuel_used_kg?: number;
  fuel_margin_kg?: number;
  max_tracking_err_m?: number;
  replan_hz?: number;
  n_replans?: number;
  n_replan_failures?: number;
  n_replans_relaxed?: number;
  mpc_solves?: number;
  mpc_failures?: number;
  mpc_mean_solve_ms?: number;
  wall_time_s?: number;
  truth?: Truth;
  [key: string]: unknown;
}

/** One closed-loop simulation record. */
export interface Sim {
  name: string;
  t: number[];
  r: Vec3[];
  v: Vec3[];
  m: number[];
  T: Vec3[];
  throttle: number[];
  r_ref?: Vec3[];
  att?: Vec3[];
  gimbal?: Vec3[];
  wind?: Vec3[];
  metrics: Metrics;
  replans?: Replan[];
  plan?: Plan;
  solved_at?: string;
}

export interface Results {
  solved_at?: string;
  scenario: Scenario;
  rocket: Rocket;
  plans: Record<string, Plan>;
  sims: Sim[];
}

export interface SolveProgress {
  stage: "plan" | "sim" | "done" | string;
  label: string;
  done: number;
  total: number;
}

/** `GET /api/status` (and the body of `POST /api/solve` / `/api/cancel`). */
export interface SolverStatus {
  running: boolean;
  version: number;
  cases?: string[] | null;
  progress?: SolveProgress | null;
  error?: string | null;
  log?: string[];
  started?: number | null;
  finished?: number | null;
  data?: string;
  data_mtime?: string | null;
}
