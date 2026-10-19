"""Stop this rental at a recorded deadline; retain disk and emergency backup.

Uses the host-provided instance-scoped key without writing it to files/logs.
This never destroys the instance. Normal completion should export data and stop
earlier; the guard protects credit if the controlling session disappears.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def write_state(state):
    path = ROOT / "logs/budget_guard_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-hours", type=float)
    args = ap.parse_args()
    policy_path = ROOT / "logs/budget_guard_policy.json"
    if args.arm_hours is not None:
        if not 0 < args.arm_hours <= 20:
            raise ValueError("Budget guard may allow at most 20 additional hours")
        policy = {"deadline_epoch": time.time() + args.arm_hours * 3600,
                  "remaining_hours_at_arm": args.arm_hours, "gpu_rate_usd_per_hour": 0.116,
                  "confirmed_credit_usd": 5.20, "action": "stop this instance, preserve disk"}
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        print("Budget deadline recorded", datetime.fromtimestamp(policy["deadline_epoch"], timezone.utc).isoformat(), flush=True)
        return
    key, instance = os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
    if not key or not instance:
        raise RuntimeError("Instance-scoped stop credentials unavailable; guard cannot be armed")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    state = {"status": "armed", "instance_id": instance, **policy}
    write_state(state)
    while time.time() < policy["deadline_epoch"]:
        completion = ROOT / "logs/full_run_state.json"
        if completion.exists():
            finished = json.loads(completion.read_text(encoding="utf-8"))
            if finished.get("status") in {"passed", "failed"}:
                grace_deadline = datetime.fromisoformat(finished["updated_utc"]).timestamp() + 30 * 60
                if time.time() >= grace_deadline:
                    state["stop_reason"] = "full job finished; 30-minute export/repair grace expired"
                    break
        time.sleep(min(60, max(0, policy["deadline_epoch"] - time.time())))
        write_state(state)
    state["status"] = "stop_due"
    write_state(state)
    subprocess.run(["supervisorctl", "stop", "pa2_full"], check=False, timeout=90)
    backup = ROOT.parent / "pa2-budget-stop-backup.tar.gz"
    proc = subprocess.run(["tar", "-czf", str(backup), "--exclude=__pycache__", "--exclude=*.pyc",
                           "-C", str(ROOT.parent), ROOT.name], check=False, timeout=300)
    state["emergency_backup"] = str(backup)
    state["backup_return_code"] = proc.returncode
    if proc.returncode == 0:
        digest = hashlib.sha256()
        with backup.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        state["emergency_backup_sha256"] = digest.hexdigest()
    state["status"] = "requesting_instance_stop"
    write_state(state)
    for _ in range(3):
        try:
            proc = subprocess.run(["vastai", "stop", "instance", instance, "--api-key", key],
                                  capture_output=True, text=True, timeout=60)
            state["stop_return_code"] = proc.returncode
            state["stop_output"] = (proc.stdout + proc.stderr).replace(key, "[REDACTED]")[-1500:]
            state["status"] = "stop_requested" if proc.returncode == 0 else "stop_failed"
        except subprocess.TimeoutExpired:
            state["status"] = "stop_failed"
            state["stop_return_code"] = -1
            state["stop_output"] = "Instance stop request timed out"
        write_state(state)
        if state["stop_return_code"] == 0:
            return
        time.sleep(10)
    raise RuntimeError("Instance stop failed; inspect budget_guard_state.json")


if __name__ == "__main__":
    main()
