"""Plan (G-FOLD + SCvx), simulate the configured cases, and export data for the 3D viewer.

Guidance, control and the simulation all run in Rust (`sim/`, Python module `boosted_sim`); this
script only drives runs, merges results and serves the three.js viewer.

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
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import boosted_sim


def solve(scenario_path: str | Path, only: list[str] | None = None, existing: dict | None = None,
          verbose: bool = True, progress=None, artifacts: Path | None = None) -> dict:
    """Plan and simulate the scenario's cases (all, or only those named in `only`) with the Rust
    flight software + sim (`boosted_sim`) and return the viewer data. With `only` and `existing`,
    the new runs are merged into the existing results; each sim record carries the plan it flew,
    so a partial re-run stays self-consistent. `progress(stage, done, total, label)` is called
    before each plan / sim. With `artifacts`, each sim's telemetry is written to
    artifacts/<mission>/<case>.telemetry.json."""
    data = json.loads(boosted_sim.solve(str(scenario_path), only=only, verbose=verbose,
                                        artifacts=str(artifacts) if artifacts else None, progress=progress))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for s in data["sims"]:
        s["solved_at"] = now
    data["solved_at"] = now
    if only is not None and existing:
        case_order = [c["name"] for c in json.loads(Path(scenario_path).read_text()).get("sim_cases", [])]
        merged = {s["name"]: s for s in existing.get("sims", [])} | {s["name"]: s for s in data["sims"]}
        data["sims"] = [merged[n] for n in case_order if n in merged]
        data["plans"] = {**existing.get("plans", {}), **data["plans"]}
    return data


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
