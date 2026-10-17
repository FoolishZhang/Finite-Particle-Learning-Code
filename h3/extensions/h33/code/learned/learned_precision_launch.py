"""Launch exactly one frozen three-shard batch; no retries or science on import."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def safe(name):
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("outside repository")
    return p


def write(p, value):
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def launch(master_path, out):
    master = json.loads(master_path.read_text(encoding="utf-8"))
    assert master["schema"] == "learned_precision_batch_v1"
    assert master["frozen"] and master["concurrency"] == 3
    assert master["timeout_seconds"] == 7260
    assert len(master["shards"]) == 3
    assert {x["rep"] for x in master["shards"]} == {0, 1, 2}
    assert sha(Path(__file__)) == master["launcher_sha256"]
    assert sha(safe("script/learned_precision_run.py")) == master["driver_sha256"]
    assert not out.exists()
    for row in master["shards"]:
        assert sha(safe(row["contract"])) == row["contract_sha256"]
        assert not safe(row["output"]).exists()
        for suffix in (".stdout.txt", ".stderr.txt", ".LAUNCH.json"):
            assert not safe(row["log_prefix"] + suffix).exists()
    started = utc()
    write(out.with_name(out.stem + "_START.json"), {
        "started_utc": started, "master_sha256": sha(master_path),
        "launcher_sha256": sha(Path(__file__)), "no_retry": True})

    def one(row):
        prefix = safe(row["log_prefix"])
        begin, clock = utc(), time.perf_counter()
        env = dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        cmd = [sys.executable, "-B", str(safe("script/learned_precision_run.py")),
               "--contract", str(safe(row["contract"])), "--out", str(safe(row["output"]))]
        timed_out = False
        with Path(str(prefix) + ".stdout.txt").open("xb") as stdout, Path(str(prefix) + ".stderr.txt").open("xb") as stderr:
            try:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                      timeout=master["timeout_seconds"], check=False)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out, code = True, None
        result = {"rep": row["rep"], "contract": row["contract"],
                  "contract_sha256": row["contract_sha256"], "output": row["output"],
                  "started_utc": begin, "finished_utc": utc(), "wall_seconds": time.perf_counter()-clock,
                  "exit_code": code, "external_timeout": timed_out, "automatic_retry": False,
                  "stdout": str(prefix.relative_to(ROOT)).replace("\\", "/") + ".stdout.txt",
                  "stderr": str(prefix.relative_to(ROOT)).replace("\\", "/") + ".stderr.txt"}
        write(Path(str(prefix) + ".LAUNCH.json"), result)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, master["shards"]))
    record = {"master_sha256": sha(master_path), "started_utc": started, "finished_utc": utc(),
              "concurrency": 3, "single_BLAS_OMP_thread": True, "shards": results,
              "all_success": all(x["exit_code"] == 0 for x in results)}
    write(out, record)
    print(json.dumps(record))
    return 0 if record["all_success"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.master.resolve(), args.out.resolve()))
