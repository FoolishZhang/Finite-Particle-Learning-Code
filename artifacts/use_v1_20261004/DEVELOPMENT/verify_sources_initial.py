"""USE0 source and contract audit only; no scientific imports or model execution."""
from pathlib import Path
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

def sha(data):
    return hashlib.sha256(data).hexdigest()

contract_sha = sha((DOC / "CONTRACT.md").read_bytes())
assert contract_sha == EXPECTED_CONTRACT, "Frozen contract changed"
manifest = json.loads((SOURCE / "MANIFEST.json").read_text(encoding="utf-8-sig"))
records = []
for row in manifest:
    path = ROOT / row["path"]
    data = path.read_bytes()
    record = {"path": row["path"], "bytes": len(data), "sha256": sha(data),
              "manifest_bytes": row["bytes"], "manifest_sha256": row["sha256"]}
    record["passed"] = record["bytes"] == row["bytes"] and record["sha256"] == row["sha256"]
    records.append(record)
assert all(row["passed"] for row in records), "Frozen source manifest mismatch"
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
    "schema_version": "1.0", "stage": "USE0", "local_date": "2026-10-04",
    "audit_completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "contract_sha256": contract_sha,
    "upstream_commit": COMMIT,
    "local_manifest_records": len(records), "local_manifest_pass": all(row["passed"] for row in records),
    "source_files": records, "remote_byte_checks": remote_rows,
    "paper": {"url": "https://arxiv.org/html/2406.19272v2", "version": "v2, 17 October 2024",
              "sections_used": ["3.1, equations (1)-(6)", "3.2", "3.3, scope only"],
              "access": "web tool HTML read; no paper file or numerical output created"},
    "scientific_budget_used": {"rng": 0, "data_processing": 0, "scoring_or_api_forward": 0, "integration": 0, "fitting_or_training": 0, "statistical_inference": 0},
    "allowed_actions": ["read source text", "hash files", "fetch fixed official text for byte comparison"],
}
(OUT / "SOURCE_AUDIT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"local_manifest_pass": receipt["local_manifest_pass"], "records": len(records),
                  "remote_fetched": sum(x["fetched"] for x in remote_rows),
                  "remote_exact_match": sum(x.get("exact_bytes_match", False) for x in remote_rows),
                  "contract_sha256": contract_sha, "scientific_budget_used": receipt["scientific_budget_used"]}, ensure_ascii=False))
