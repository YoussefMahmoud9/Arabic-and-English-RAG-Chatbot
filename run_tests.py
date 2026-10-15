"""One-command test runner.

    python run_tests.py              # unit + live (live skipped if app is down)
    python run_tests.py --unit       # fast checks only, no server needed
    python run_tests.py --live       # only the tests that need the running app
    python run_tests.py -k arabic    # extra args are passed straight to pytest

Exit code is non-zero if anything fails, so it works in CI.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
APP_URL = "http://127.0.0.1:7860"


def app_is_up(timeout: float = 2.0) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{APP_URL}/", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run(label: str, args: list[str]) -> tuple[str, bool, float]:
    print(f"\n{'=' * 62}\n {label}\n{'=' * 62}", flush=True)
    start = time.time()
    proc = subprocess.run([PYTHON, "-m", "pytest", *args], cwd=str(ROOT))
    return label, proc.returncode == 0, time.time() - start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", action="store_true", help="only fast, offline checks")
    ap.add_argument("--live", action="store_true", help="only tests needing the app")
    ap.add_argument("-v", "--verbose", action="store_true")
    args, extra = ap.parse_known_args()

    common = ["-ra", "--strict-markers"]
    if args.verbose:
        common.append("-vv")
    else:
        common.append("-q")

    stages: list[tuple[str, list[str]]] = []
    both = not (args.unit or args.live)

    if args.unit or both:
        stages.append(("Unit / static checks", common + ["-m", "unit", *extra]))
    if args.live or both:
        if app_is_up():
            stages.append(("Live behaviour (app running)", common + ["-m", "live", *extra]))
        elif args.live:
            print(f"[ERROR] app is not running at {APP_URL}.")
            print("        Start it with start_rag.bat, then retry.")
            return 2
        else:
            print(f"[SKIP] live tests: app not running at {APP_URL}")
            print("       (start it with start_rag.bat to include them)")

    if not stages:
        print("[ERROR] nothing selected")
        return 2

    results = []
    for label, cmd in stages:
        results.append(run(label, cmd))

    print(f"\n{'=' * 62}\n Summary\n{'=' * 62}")
    for label, ok, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:32s} {secs:6.1f}s")
    failed = [label for label, ok, _ in results if not ok]
    if failed:
        print(f"\n  {len(failed)} stage(s) failed: {', '.join(failed)}")
        return 1
    print("\n  All selected stages passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())