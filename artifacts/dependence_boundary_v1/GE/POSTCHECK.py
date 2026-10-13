"""Read-only endpoint reconstruction; writes a separate software audit receipt."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def run():
    rows = list(csv.DictReader((HERE / "IMAGE_CONTRASTS.csv").open(encoding="utf-8")))
    primary = list(csv.DictReader((HERE / "PRIMARY.csv").open(encoding="utf-8")))
    names = [r["contrast"] for r in primary]
    values = np.array([[float(r[name]) for name in names] for r in rows])
    groups, inverse = np.unique([r["group_id"] for r in rows], return_inverse=True)
    totals = np.zeros((len(groups), 16)); counts = np.zeros(len(groups), dtype=int)
    for group, value in zip(inverse, values):
        totals[group] += value; counts[group] += 1
    rng = np.random.default_rng(20269811)
    samples = []
    for _ in range(50):
        chosen = rng.integers(len(groups), size=(200, len(groups)))
        samples.append(totals[chosen].sum(axis=1) / counts[chosen].sum(axis=1)[:, None])
    samples = np.concatenate(samples)
    lo, hi = np.quantile(samples, [.05 / 32, 1 - .05 / 32], axis=0)
    mean_error = float(np.max(abs(values.mean(0) - [float(r["mean"]) for r in primary])))
    interval_error = float(max(np.max(abs(lo - [float(r["lower"]) for r in primary])),
                               np.max(abs(hi - [float(r["upper"]) for r in primary]))))
    assert max(mean_error, interval_error) < 1e-12
    pool_error = 0.; min_jensen_gap = float("inf"); finite_files = 0
    for path in sorted((HERE / "scores").glob("*/*.npz")):
        with np.load(path) as archive:
            pooled = archive["pooled_scores"]
            streams = archive["stream_scores"]
            assert all(np.isfinite(archive[k]).all() for k in archive.files)
            # NP/EQ mixture target probability can be reconstructed from each
            # stream's target NLL, independently of the inference implementation.
            reconstructed = -np.log(np.exp(-streams[:, :, :2, 0]).mean(axis=0))
            pool_error = max(pool_error, float(np.max(abs(reconstructed - pooled[:, :2, 0]))))
            min_jensen_gap = min(min_jensen_gap, float(np.min(streams[:, :, :, 0].mean(0) - pooled[:, :, 0])))
            finite_files += 1
    assert finite_files == 48 and pool_error < 1e-10 and min_jensen_gap > -1e-10
    receipt = dict(
        time_utc=datetime.now(timezone.utc).isoformat(), status="pass_independent_public_statistics_and_probability_pool_reconstruction",
        public_images=len(rows), public_groups=len(groups), primary_contrasts=len(primary),
        bootstrap_draws=10000, bootstrap_mean_max_error=mean_error,
        bootstrap_interval_max_error=interval_error, local_score_files=finite_files,
        NP_EQ_probability_pool_max_error=pool_error, minimum_stream_mean_minus_pooled_NLL=min_jensen_gap,
        endpoint_or_rule_changes=False, original_complete_sha256=hashlib.sha256((HERE / "COMPLETE.json").read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output = HERE / "POSTCHECK.json"
    assert not output.exists()
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(receipt)


if __name__ == "__main__":
    run()
