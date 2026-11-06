"""Plan (G-FOLD + SCvx), simulate the configured cases, and export data for the 3D viewer.

    uv run python run.py [scenario.json]            solve all cases, write web/data/results.json
    uv run python run.py --only "CASE NAME"         re-solve one case, merged into the stored results
    uv run python run.py --serve                    open the viewer on the stored results (no solve);
                                                    solve all / the selected case from the app
    uv run python run.py --serve --solve            solve all, then open the viewer
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
import subprocess
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

import numpy as np

from landing.dynamics import TruthModel, open_loop_resim
from landing.guidance import GFoldGuidance, SCvxGuidance
from landing.ignition import plan_with_ignition
from landing.sim import simulate, truth_from_case
from landing.vehicle import Scenario

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def solve(scenario_path: str | Path, only: list[str] | None = None, existing: dict | None = None,
          verbose: bool = True, progress=None, artifacts: Path | None = None) -> dict:
    """Plan and simulate the scenario's cases (all, or only those named in `only`) and return the
    viewer data. With `only` and `existing`, the new runs are merged into the existing results.
    Each sim record carries the plan it flew, so a partial re-run stays self-consistent.
    `progress(stage, done, total, label)` is called before each plan / sim.
    With `artifacts`, each sim's plant telemetry is written to artifacts/<mission>/<case>.telemetry.json."""
    sc = Scenario.from_json(scenario_path)
    rk = sc.rocket
    print(f"Scenario: {sc.name} | rocket: {rk.name}")
    cases = list(sc.sim_cases)
    if only is not None:
        missing = [n for n in only if n not in {c["name"] for c in cases}]
        if missing:
            raise ValueError(f"cases not in {scenario_path}: {missing}")
        cases = [c for c in cases if c["name"] in only]
    need = {c["plan"] for c in cases}
    if "scvx" in need and not sc.coast:
        need.add("gfold")  # SCvx is warm-started from G-FOLD without a coast phase
    keys = [k for k in ("gfold", "scvx") if k in need]
    total, done = len(keys) + len(cases), 0

    def step(stage: str, label: str) -> None:
        nonlocal done
        if progress:
            progress(stage, done, total, label)
        done += 1

    guidance = {"gfold": GFoldGuidance(sc), "scvx": SCvxGuidance(sc)}
    plans = {}
    for key in keys:
        step("plan", {"gfold": "G-FOLD plan", "scvx": "SCvx plan"}[key])
        t0 = time.perf_counter()
        if sc.coast:  # engine-off start: choose ignition time, then plan from the predicted state
            p = plan_with_ignition(sc, guidance[key], verbose=verbose)
        else:
            p = guidance[key].plan(verbose=verbose, **({"init": plans["gfold"]} if key == "scvx" else {}))
        p.info["solve_time_s"] = time.perf_counter() - t0
        resim = open_loop_resim(TruthModel(rk, sc.env, drag=key == "scvx"), p)
        p.info["open_loop_terminal_err"] = [float(np.linalg.norm(resim[0:3])), float(np.linalg.norm(resim[3:6]))]
        plans[key] = p

    print("\nPlans")
    print(f"  {'':8s} {'ignite':>7s} {'h_ig':>7s} {'burn':>7s} {'m_f [kg]':>10s} {'fuel [kg]':>10s} {'tight':>9s} {'resim r/v err':>16s}")
    for k, p in plans.items():
        e = p.info["open_loop_terminal_err"]
        print(
            f"  {k:8s} {p.info.get('t_ignition', 0.0):6.2f}s {p.r[0, 0]:6.0f}m {p.tf:6.2f}s {p.m[-1]:10.1f} {sc.m0 - p.m[-1]:10.1f} "
            f"{p.info['tightness_max']:9.1e} {e[0]:7.4f}m {e[1]:6.4f}m/s"
        )
    plan_dicts = {k: p.to_dict(rk.T_max) for k, p in plans.items()}

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    new_sims = {}
    print("\nClosed-loop simulations")
    print(f"  {'case':62s} {'pos err':>8s} {'v_vert':>7s} {'v_hor':>6s} {'tilt':>5s} {'fuel':>7s} {'gsat%':>5s} {'replans':>7s}")
    for case in cases:
        step("sim", case["name"])
        t0 = time.perf_counter()
        res = simulate(
            sc,
            plans[case["plan"]],
            truth_from_case(sc, case.get("truth", {})),
            case["name"],
            guidance=guidance[case["plan"]],
            replan_hz=case.get("replan_hz", 0.0),
            controller=case.get("controller", "cascade"),
            telemetry_path=artifacts / Path(scenario_path).parent.name / f"{slug(case['name'])}.telemetry.json"
            if artifacts else None,
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
        new_sims[case["name"]] = {**res.to_dict(rk.T_max), "plan": plan_dicts[case["plan"]], "solved_at": now}
    if progress:
        progress("done", total, total, "")

    old = {s["name"]: s for s in (existing or {}).get("sims", [])} if only is not None else {}
    sims = [new_sims.get(c["name"]) or old.get(c["name"]) for c in sc.sim_cases]
    return {
        "solved_at": now,
        "scenario": {
            "name": sc.name,
            "path": str(scenario_path),
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
            "attitude": {k: v for k, v in (rk.attitude or {}).items() if not k.startswith("_")},
        },
        "plans": {**(existing or {}).get("plans", {}), **plan_dicts} if only is not None else plan_dicts,
        "sims": [s for s in sims if s is not None],
    }


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def write_results(data: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=float, ensure_ascii=False))
    tmp.replace(out)  # atomic: the viewer never reads a half-written file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="missions/coast_landing/scenario.json")
    ap.add_argument("--out", default="web/data/results.json")
    ap.add_argument("--serve", action="store_true", help="open the viewer on the stored results (solves first only if there are none)")
    ap.add_argument("--solve", action="store_true", help="with --serve: re-solve before serving")
    ap.add_argument("--only", action="append", metavar="CASE", help="solve only this case (repeatable); merged into existing results")
    ap.add_argument("--artifacts", default="artifacts", help="directory for per-run telemetry artifacts ('' = none)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--progress-json", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    if not args.serve or args.solve or not out.exists():
        existing = json.loads(out.read_text()) if args.only and out.exists() else None
        progress = None
        if args.progress_json:
            def progress(stage, done, total, label):
                print("@@progress " + json.dumps({"stage": stage, "done": done, "total": total, "label": label}), flush=True)
        data = solve(args.scenario, only=args.only, existing=existing, verbose=not args.quiet, progress=progress,
                     artifacts=Path(args.artifacts) if args.artifacts else None)
        write_results(data, out)
        print(f"\nWrote {out}")
    if args.serve:
        serve(args.port, out, args.scenario)
    else:
        print("View it with:  uv run python run.py --serve")


class SolveJob:
    """One background solve at a time, run as a subprocess (`run.py --only ... --progress-json`)
    so the solver never blocks the web server. Progress and log lines are parsed from its stdout."""

    def __init__(self, scenario: str, out: Path):
        self.scenario, self.out = scenario, out
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.state = {"running": False, "progress": None, "log": [], "error": None, "version": 0,
                      "cases": None, "started": None, "finished": None}

    def status(self) -> dict:
        with self.lock:
            return {**self.state, "log": self.state["log"][-40:]}

    def start(self, cases: list[str] | None) -> bool:
        with self.lock:
            if self.state["running"]:
                return False
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), self.scenario, "--out", str(self.out),
                   "--progress-json", "-q"]
            for c in cases or []:
                cmd += ["--only", c]
            self.proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.state.update(running=True, progress=None, log=[], error=None, cases=cases,
                              started=time.time(), finished=None)
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
        return True

    def cancel(self) -> None:
        with self.lock:
            if self.proc and self.state["running"]:
                self.proc.terminate()

    def _watch(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = line.rstrip()
            with self.lock:
                if line.startswith("@@progress "):
                    self.state["progress"] = json.loads(line[len("@@progress "):])
                elif line:
                    self.state["log"].append(line)
                    print(f"  [solve] {line}")
        rc = proc.wait()
        with self.lock:
            self.state.update(running=False, finished=time.time())
            if rc == 0:
                self.state["version"] += 1
            else:
                tail = [l for l in self.state["log"] if l.strip()][-6:]
                self.state["error"] = "cancelled" if rc < 0 else ("\n".join(tail) or f"solver exited with code {rc}")


def serve(port: int, data_path: Path, scenario: str) -> None:
    web = Path(__file__).resolve().parent / "web"
    rel = data_path.resolve().relative_to(web).as_posix()
    job = SolveJob(scenario, data_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(web), **kw)

        def end_headers(self):  # always serve the latest viewer code / results
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt, *a):  # keep the console for solver output
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/status"):
                mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(data_path.stat().st_mtime)) if data_path.exists() else None
                return self._json({**job.status(), "data": rel, "data_mtime": mtime})
            super().do_GET()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/solve":
                cases = req.get("cases")  # None = all
                if not job.start(cases if isinstance(cases, list) else None):
                    return self._json({"error": "a solve is already running"}, 409)
                return self._json(job.status())
            if self.path == "/api/cancel":
                job.cancel()
                return self._json(job.status())
            self._json({"error": "not found"}, 404)

    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/index.html?data={rel}"
        print(f"Serving viewer at {url}  (Ctrl+C to stop)")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
