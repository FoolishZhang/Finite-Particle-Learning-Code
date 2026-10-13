"""USE0 sources: --check is offline/read-only; --run creates a fresh receipt once."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/use_v1_20261004"
DOC = ROOT / "docs/08_research_研究工作/USE_v1_20261004"
SOURCE = ROOT / "reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a"
COMMIT = "58c3e602cb11fb4ea9106226e0f9e58f50d5d10a"
EXPECTED_CONTRACT = "a84780325e5205a27f1e066387ae663f4edb877486766339214072459e2ade3b"
ZERO_BUDGET = {"rng": 0, "data_processing": 0, "scoring_or_api_forward": 0, "integration": 0, "fitting_or_training": 0, "statistical_inference": 0}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local_sources():
    contract_sha = sha((DOC / "CONTRACT.md").read_bytes())
    if contract_sha != EXPECTED_CONTRACT:
        raise RuntimeError("Frozen contract changed")
    manifest = json.loads((SOURCE / "MANIFEST.json").read_text(encoding="utf-8-sig"))
    records = []
    for row in manifest:
        data = (ROOT / row["path"]).read_bytes()
        record = {"path": row["path"], "bytes": len(data), "sha256": sha(data),
                  "manifest_bytes": row["bytes"], "manifest_sha256": row["sha256"]}
        record["passed"] = record["bytes"] == row["bytes"] and record["sha256"] == row["sha256"]
        records.append(record)
    if not all(row["passed"] for row in records):
        raise RuntimeError("Frozen source manifest mismatch")
    return contract_sha, records


def check():
    contract_sha, records = local_sources()
    receipt = json.loads((OUT / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    if receipt["contract_sha256"] != contract_sha or receipt["source_files"] != records:
        raise RuntimeError("Receipt and current local source bytes disagree")
    if receipt["scientific_budget_used"] != ZERO_BUDGET:
        raise RuntimeError("Unexpected scientific budget")
    for row in receipt["remote_byte_checks"]:
        current_sha = sha((ROOT / row["path"]).read_bytes())
        if row["local_sha256"] != current_sha:
            raise RuntimeError("Remote receipt local hash mismatch")
        if row.get("exact_bytes_match") and row.get("remote_sha256") != current_sha:
            raise RuntimeError("Remote receipt inconsistent hash assertion")
    print(json.dumps({"mode": "check", "read_only": True, "network_requests": 0,
                      "local_manifest_pass": True, "records": len(records),
                      "receipt_sha256": sha((OUT / "SOURCE_AUDIT.json").read_bytes()),
                      "contract_sha256": contract_sha, "scientific_budget_used": ZERO_BUDGET}))


def run():
    receipt_path = OUT / "SOURCE_AUDIT.json"
    if receipt_path.exists():
        raise RuntimeError("SOURCE_AUDIT.json already exists; --run refuses overwrite. Use --check.")
    contract_sha, records = local_sources()
    remote_rows = []
    for name in ["models/models.py", "models/losses.py", "utils/training.py", "configs/model/SCBM.yaml", "configs/model/model_defaults.yaml", "README.md"]:
        url = f"https://raw.githubusercontent.com/mvandenhi/SCBM/{COMMIT}/{name}"
        local = (SOURCE / name).read_bytes()
        row = {"path": str((SOURCE / name).relative_to(ROOT)).replace("\\", "/"),
               "url": url, "local_sha256": sha(local)}
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                remote = response.read()
            row.update(remote_sha256=sha(remote), remote_bytes=len(remote),
                       exact_bytes_match=remote == local, fetched=True)
        except Exception as exc:
            row.update(fetched=False, error=type(exc).__name__ + ": " + str(exc))
        remote_rows.append(row)
    receipt = {
        "schema_version": "1.1", "stage": "USE0", "local_date": "2026-10-04",
        "audit_completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "contract_sha256": contract_sha, "upstream_commit": COMMIT,
        "local_manifest_records": len(records), "local_manifest_pass": True,
        "source_files": records, "remote_byte_checks": remote_rows,
        "paper": {"url": "https://arxiv.org/html/2406.19272v2", "version": "v2, 17 October 2024",
                  "sections_used": ["3.1, equations (1)-(6)", "3.2", "3.3, scope only"],
                  "access": "web tool HTML read; no paper file or numerical output created"},
        "scientific_budget_used": ZERO_BUDGET,
        "allowed_actions": ["read source text", "hash files", "fetch fixed official text for byte comparison"],
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mode": "run", "receipt_created": True, "records": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Offline, read-only verification of source bytes and existing receipt")
    mode.add_argument("--run", action="store_true", help="Fetch fixed upstream text and create a receipt; refuses any overwrite")
    args = parser.parse_args()
    check() if args.check else run()
