"""
List (and optionally stop) leftover Modal apps from this project.

    python -m harness.cleanup            # dry run: show running optimization-agent apps
    python -m harness.cleanup --stop     # stop them (terminates their containers; do not run during a live loop)

A normal run needs none of this: the loop's app is stopped when it exits. Leftovers come from
abnormal exits (kill -9, closed terminal) and are otherwise only reaped by Modal's heartbeat timeout.
"""
import argparse
import json
import subprocess
import sys

APP_NAME = "optimization-agent"


def _modal(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True)


def running_apps() -> list[dict]:
    r = _modal("app", "list", "--json")
    if r.returncode:
        raise SystemExit(f"modal app list failed: {r.stderr.strip()[-300:]}")
    return [a for a in json.loads(r.stdout) if a["description"] == APP_NAME and a["state"] != "stopped"]


def stop(app_id: str) -> bool:
    r = _modal("app", "stop", "--yes", app_id)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="actually stop the apps (default: dry run)")
    a = ap.parse_args()
    apps = running_apps()
    if not apps:
        print(f"no running '{APP_NAME}' apps on Modal; nothing to clean up")
        return
    for app in apps:
        print(f"{app['app_id']}  state={app['state']} tasks={app['tasks']} created={app['created_at']}")
        if a.stop:
            print("  stopped" if stop(app["app_id"]) else "  FAILED to stop (see `modal app stop`)")
    if not a.stop:
        print("dry run: re-run with --stop to stop these (do not do that while a loop is running)")


if __name__ == "__main__":
    main()
