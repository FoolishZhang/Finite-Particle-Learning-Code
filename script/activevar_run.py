"""Bounded orchestration of the already-frozen activevar producer.

No scientific implementation lives here. Reference is run by the parent and
must qualify. Every MC stream is dispatched at most once, with at most four
processes, followed by one summary only if every stream succeeded.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/active_scale_boundary"
EXPECTED_FREEZE = "5b720b19a12def0fb5446a40bb3ce35ee16aa033ecdcb17704a9de7754e6b23c"
LOCK = threading.Lock()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def log(message):
    with LOCK:
        with (OUT / "RUN_CONSOLE.log").open("a", encoding="utf8", newline="\n") as handle:
            handle.write(message + "\n")
        print(message, flush=True)


def invoke(arguments):
    started = stamp()
    log(json.dumps({"event": "start", "arguments": arguments, "utc": started}))
    run = subprocess.run([sys.executable, "script/activevar_compute.py", *arguments],
                         cwd=ROOT, text=True, encoding="utf8", errors="replace",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    record = {"arguments": arguments, "started_utc": started, "finished_utc": stamp(),
              "returncode": run.returncode, "console": run.stdout}
    log(json.dumps(record, ensure_ascii=False))
    return record


def main():
    if sha(OUT / "FREEZE.json") != EXPECTED_FREEZE:
        raise RuntimeError("Unexpected science freeze")
    frozen = json.loads((OUT / "FREEZE.json").read_text(encoding="utf8"))
    if sha(ROOT / "script/activevar_compute.py") != frozen["source_sha256"]:
        raise RuntimeError("Frozen producer changed")
    if (OUT / "MC").exists() or (OUT / "SUMMARY").exists():
        raise RuntimeError("An MC or summary unit already exists; runner will not repeat it")
    with (OUT / "RUN_CONSOLE.log").open("x", encoding="utf8"):
        pass
    save(OUT / "RUNNER.json", {"started_utc": stamp(), "runner_sha256": sha(__file__),
                                "freeze_sha256": EXPECTED_FREEZE, "maximum_processes": 4,
                                "streams": list(range(16)), "reference_owned_by_parent": True,
                                "failure_policy": "Finish already-dispatched units; stop new dispatch and summary; never retry."})
    wait_start = time.monotonic()
    while True:
        if (OUT / "REFERENCE/FAILURE.json").exists():
            log("REFERENCE failed; no dependent science dispatched.")
            save(OUT / "RUNNER_COMPLETE.json", {"finished_utc": stamp(), "success": False,
                                                "reason": "reference failure", "dispatched_streams": []})
            return 1
        path = OUT / "REFERENCE/RESULTS.json"
        if path.exists():
            try:
                reference = json.loads(path.read_text(encoding="utf8"))
            except json.JSONDecodeError:
                time.sleep(2.)
                continue
            if not reference["qualification_passed"]:
                log("REFERENCE qualification failed; no dependent science dispatched.")
                save(OUT / "RUNNER_COMPLETE.json", {"finished_utc": stamp(), "success": False,
                                                    "reason": "reference qualification", "dispatched_streams": []})
                return 1
            if reference["freeze_sha256"] != EXPECTED_FREEZE:
                raise RuntimeError("Reference belongs to a different freeze")
            break
        if time.monotonic() - wait_start > 3600:
            raise TimeoutError("Parent reference did not finish within the orchestration wait limit")
        time.sleep(5.)
    log(json.dumps({"event": "reference_qualified", "reference_sha256": sha(path), "utc": stamp()}))
    completed, dispatched, failures = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {}
        next_stream = 0
        while next_stream < 16 or pending:
            while next_stream < 16 and len(pending) < 4 and not failures:
                stream = next_stream
                future = pool.submit(invoke, ["mc", "--stream", str(stream)])
                pending[future] = stream
                dispatched.append(stream)
                next_stream += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                stream = pending.pop(future)
                record = future.result()
                completed.append(record)
                if record["returncode"] != 0:
                    failures.append(stream)
    summary = None
    if len(dispatched) == 16 and not failures:
        summary = invoke(["summarize"])
    success = len(dispatched) == 16 and not failures and summary is not None and summary["returncode"] == 0
    save(OUT / "RUNNER_COMPLETE.json", {"finished_utc": stamp(), "success": success,
                                        "dispatched_streams": dispatched, "failed_streams": failures,
                                        "completed": completed, "summary": summary,
                                        "runner_sha256": sha(__file__), "freeze_sha256": EXPECTED_FREEZE})
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
