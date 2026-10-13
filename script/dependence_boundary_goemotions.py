"""Restore only GoEmotions and diagnose fixed-model finite-K covariance boundaries."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import unicodedata
import urllib.request
import warnings

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/dependence_boundary_v1/GDATA"
PLAN = ROOT / "docs/08_research_研究工作/边际依赖接口_20260929/PLAN_GDATA.md"
LEGACY = ROOT / "artifacts/cbm_public/reference_source_v5_20260908"
DATA = ROOT / "data/GoEmotions_reader_release_v5"
INPUTS = [PLAN, Path(__file__), DATA / "MANIFEST.json", DATA / "processed/RECORD_MANIFEST.csv",
          DATA / "processed/D0_AUDIT.json", LEGACY / "EXECUTION_FREEZE.json",
          ROOT / "script/cbm_public_v5_data.py", ROOT / "script/cbm_public_v5_repartition.py",
          ROOT / "script/cbm_public_v5_fit.py"]
MODELS = {kind: LEGACY / "models/GoEmotions" / kind / "predictor.joblib" for kind in ["word", "word_char"]}
ROLES = ["train", "association_fit"]
CONFIG = dict(models=list(MODELS), roles=ROLES, scales=[.5, 1., 2.], raters=[0, 1, 2], primary_rater=0,
              K=[1, 2, 4, 10, 16, 100], concepts=27, combinations=36, derivative_rows=216,
              prediction_max_error=1e-10, matrix_atol=1e-11, matrix_rtol=1e-11,
              derivative_max_error=1e-11, preferred_sklearn="1.9.0", fit_new_models=False)


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def norm(s): return " ".join(unicodedata.normalize("NFKC", s).casefold().split())
def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
def csvwrite(p, rows):
    with p.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def freeze():
    assert not (OUT / "FREEZE.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitignore").write_text("/raw/\n/cache/\n/env/\n*.npz\n", encoding="utf-8", newline="\n")
    sources = [x for x in read(DATA / "MANIFEST.json")["acquired_sources"]
               if Path(x["path"]).name in ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv", "emotions.txt"]]
    assert len(sources) == 4
    legacy = read(LEGACY / "EXECUTION_FREEZE.json")["fitted_assets"]
    for p in MODELS.values(): assert sha(p) == legacy[p.relative_to(ROOT).as_posix()]
    write(OUT / "FREEZE.json", dict(time_utc=now(), config=CONFIG, sources=sources,
          files={p.relative_to(ROOT).as_posix(): sha(p) for p in INPUTS + list(MODELS.values())},
          no_new_training=True, no_clinical_assets=True, no_test_prediction_or_scoring=True))


def check_freeze():
    f = read(OUT / "FREEZE.json"); assert f["config"] == CONFIG
    for p, h in f["files"].items(): assert sha(ROOT / p) == h, p
    return f


def environment():
    check_freeze()
    versions = {}
    for name in ["numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"]:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    write(OUT / "ENVIRONMENT.json", dict(time_utc=now(), executable=sys.executable,
          python=sys.version, platform=platform.platform(), packages=versions,
          matching_serialized_sklearn=versions["scikit-learn"] == "1.9.0"))
    print(versions, flush=True)


def download():
    frozen = check_freeze(); receipts = []
    for item in frozen["sources"]:
        path = OUT / "raw" / Path(item["path"]).name
        path.parent.mkdir(exist_ok=True)
        start = time.perf_counter()
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            assert not temporary.exists(), "Preserve an interrupted download rather than overwriting it"
            request = urllib.request.Request(item["source_url"], headers={"User-Agent": "TCM-SD-research-source-restoration"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
                while True:
                    block = response.read(2**20)
                    if not block: break
                    stream.write(block)
            assert sha(temporary) == item["sha256"], path.name
            temporary.rename(path)
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        receipts.append(dict(file=path.name, url=item["source_url"], sha256=sha(path), bytes=path.stat().st_size,
                             seconds=time.perf_counter() - start))
        print("verified official source", path.name, path.stat().st_size, flush=True)
    write(OUT / "DOWNLOAD.json", dict(time_utc=now(), sources=receipts, clinical_sources_accessed=False))


class Components:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i
    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b: self.parent[max(a, b)] = min(a, b)


def package():
    check_freeze(); assert (OUT / "DOWNLOAD.json").exists()
    assert not (OUT / "DATA_SEAL.json").exists()
    with (DATA / "processed/RECORD_MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    lookup = {r["id"]: r for r in manifest}; assert len(lookup) == len(manifest) == 56451
    names = [s for s in (OUT / "raw/emotions.txt").read_text(encoding="utf-8").splitlines() if s != "neutral"]
    assert len(names) == 27 and names == read(DATA / "processed/D0_AUDIT.json")["concepts"]
    records = {}; raw_rows = 0; seen_annotations = set()
    for path in sorted((OUT / "raw").glob("goemotions_*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                raw_rows += 1; item = row["id"]; rater = row["rater_id"]
                assert (item, rater) not in seen_annotations
                seen_annotations.add((item, rater))
                hashed = digest("go|" + item)[:24]; info = lookup.get(hashed)
                meta = (row["text"], row["author"], row["link_id"])
                if item not in records: records[item] = dict(id=hashed, meta=meta, clear=[])
                record = records[item]; assert record["meta"] == meta
                flag = row["example_very_unclear"].lower(); assert flag in ["true", "false"]
                if flag == "false":
                    # Non-analyzed roles never have emotion cells converted or retained.
                    label = None
                    if info is not None and info["role"] in ROLES:
                        assert all(row[c] in ["0", "1"] for c in names)
                        label = [int(row[c]) for c in names]
                    record["clear"].append((digest("20260908501|" + item + "|" + rater), rater, label))
    assert raw_rows == 211225
    eligible = []; excluded = 0
    for item in sorted(records):
        record = records[item]
        if len(record["clear"]) < 3: excluded += 1; continue
        assert record["id"] in lookup
        record["selected"] = sorted(record.pop("clear"))[:3]
        info = lookup[record["id"]]
        assert digest(norm(record["meta"][0])) == info["text_hash"]
        record["role"] = info["role"]
        eligible.append(record)
    assert len(eligible) == 56451 and excluded == 1560
    assert {r["id"] for r in eligible} == set(lookup)
    components = Components(len(eligible)); maps = {key: {} for key in ["author", "thread", "text_hash"]}
    for i, record in enumerate(eligible):
        text, author, thread = record["meta"]
        for key, value in [("author", author.casefold()), ("thread", thread.casefold()), ("text_hash", digest(norm(text)))]:
            if not value or (key == "author" and value in ["[deleted]", "[removed]", "deleted", "removed", "nan"]): continue
            if value in maps[key]: components.union(i, maps[key][value])
            else: maps[key][value] = i
    counts = Counter()
    for i, record in enumerate(eligible):
        group = "go_group_" + eligible[components.find(i)]["id"]
        assert group == lookup[record["id"]]["group_id"]
        record["group"] = group; counts[group] += 1
    all_roles = ["train", "model_calibration", "association_fit", "policy_validation", "test_locked"]
    for record in eligible:
        uniform = int(digest("20260908502|" + record["group"]), 16) / 2**256
        index = min(int(np.searchsorted([.6, .7, .8, .9, 1.], uniform, side="right")), 4)
        assigned = "train" if counts[record["group"]] > .1 * len(eligible) else all_roles[index]
        assert assigned == record["role"]
    byid = {r["id"]: r for r in eligible}; files = []; summaries = []
    (OUT / "cache").mkdir(exist_ok=True)
    for role in ROLES:
        selected = [byid[r["id"]] for r in manifest if r["role"] == role]
        ids = np.array([r["id"] for r in selected]); groups = np.array([r["group"] for r in selected])
        labels = np.array([[v[2] for v in r["selected"]] for r in selected], dtype=np.float64)
        raters = np.array([[v[1] for v in r["selected"]] for r in selected])
        assert labels.shape == (len(selected), 3, 27)
        path = OUT / "cache" / (role + ".npz")
        np.savez_compressed(path, id=ids, group_id=groups, labels=labels, rater_id=raters); files.append(path)
        path = OUT / "cache" / (role + ".jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for r in selected: stream.write(json.dumps(dict(id=r["id"], text=r["meta"][0]), ensure_ascii=False) + "\n")
        files.append(path); summaries.append(dict(role=role, records=len(selected), groups=len(set(groups))))
    assert summaries == [dict(role="train", records=33395, groups=22084), dict(role="association_fit", records=5577, groups=3777)]
    write(OUT / "DATA_AUDIT.json", dict(time_utc=now(), status="pass_frozen_metadata_and_rater_contract",
          raw_annotation_rows=raw_rows, eligible_comments=len(eligible), excluded_comments=excluded,
          all_text_hashes_and_connected_groups_match=True, all_role_hash_assignments_match=True,
          analyzed_roles=summaries, concepts=names, labels_shape="N x 3 x 27",
          primary_rater=0, sensitivity_raters=[1, 2], other_role_emotion_values_numerically_parsed_or_retained=False,
          old_test_predictions_or_outcomes_read=False))
    files.append(OUT / "DATA_AUDIT.json")
    write(OUT / "DATA_SEAL.json", dict(time_utc=now(), files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print("data restored and isolated", summaries, flush=True)


def check_seal(name):
    for p, h in read(OUT / name)["files"].items(): assert sha(ROOT / p) == h, p


def manual_tfidf(vectorizer, texts):
    from scipy.sparse import csr_matrix
    assert vectorizer.norm == "l2" and vectorizer.use_idf and vectorizer.sublinear_tf
    analyzer = vectorizer.build_analyzer(); vocabulary = vectorizer.vocabulary_
    indptr = [0]; indices = []; values = []
    for text in texts:
        counts = Counter(vocabulary[token] for token in analyzer(text) if token in vocabulary)
        columns = sorted(counts); weights = np.array([(1 + np.log(counts[j])) * vectorizer.idf_[j] for j in columns])
        denominator = np.linalg.norm(weights)
        if denominator: weights /= denominator
        indices.extend(columns); values.extend(weights.tolist()); indptr.append(len(indices))
    return csr_matrix((values, indices, indptr), shape=(len(texts), len(vocabulary)))


def predict():
    check_freeze(); check_seal("DATA_SEAL.json"); assert not (OUT / "PREDICTION_SEAL.json").exists()
    import joblib
    from scipy.sparse import hstack
    from threadpoolctl import threadpool_limits
    environment(); receipts = []; files = []
    with threadpool_limits(limits=1):
        for kind, path in MODELS.items():
            with warnings.catch_warnings(record=True) as noted:
                warnings.simplefilter("always"); predictor = joblib.load(path)
            assert predictor["kind"] == kind and len(predictor["classifiers"]) == 27
            coef = np.stack([c.coef_[0] for c in predictor["classifiers"]])
            intercept = np.array([c.intercept_[0] for c in predictor["classifiers"]])
            assert all(np.array_equal(c.classes_, [0, 1]) for c in predictor["classifiers"])
            for role in ROLES:
                start = time.perf_counter()
                rows = [json.loads(s) for s in (OUT / "cache" / (role + ".jsonl")).read_text(encoding="utf-8").splitlines()]
                texts = [r["text"] for r in rows]
                with np.load(OUT / "cache" / (role + ".npz")) as z: assert np.array_equal(z["id"], [r["id"] for r in rows])
                word = predictor["word"].transform(texts)
                x = word if kind == "word" else hstack([word, predictor["char"].transform(texts)], format="csr") / np.sqrt(2)
                direct = np.asarray(x @ coef.T) + intercept
                api = np.stack([c.decision_function(x) for c in predictor["classifiers"]], axis=1)
                probabilities = np.stack([c.predict_proba(x)[:, 1] for c in predictor["classifiers"]], axis=1)
                manual_word = manual_tfidf(predictor["word"], texts[:16])
                manual_x = manual_word if kind == "word" else hstack([manual_word, manual_tfidf(predictor["char"], texts[:16])], format="csr") / np.sqrt(2)
                diff = manual_x - x[:16]
                feature_error = float(max(abs(diff.data), default=0))
                explicit_error = float(np.max(abs(direct - api)))
                independent_error = float(np.max(abs(np.asarray(manual_x @ coef.T) + intercept - api[:16])))
                probability_error = float(np.max(abs(expit(api) - probabilities)))
                precision = max(feature_error, explicit_error, independent_error, probability_error) <= CONFIG["prediction_max_error"]
                receipt = dict(kind=kind, role=role, examples=len(rows), features=x.shape[1], sparse_nnz=x.nnz,
                               manual_feature_max_error=feature_error, explicit_linear_max_error=explicit_error,
                               independent_feature_and_linear_max_error=independent_error,
                               probability_max_error=probability_error, precision_pass=bool(precision),
                               serialized_sklearn="1.9.0", actual_sklearn=importlib.metadata.version("scikit-learn"),
                               load_warnings=[str(w.message) for w in noted], seconds=time.perf_counter() - start)
                receipts.append(receipt); write(OUT / "PREDICTION_AUDIT.json", dict(status="running" if precision else "failed_prediction_precision", rows=receipts))
                assert precision, receipt
                output = OUT / "cache" / f"{kind}_{role}_means.npz"
                np.savez_compressed(output, means=api, ids=np.asarray([r["id"] for r in rows])); files.append(output)
                print("verified frozen predictions", kind, role, len(rows), max(feature_error, explicit_error, independent_error, probability_error), flush=True)
    write(OUT / "PREDICTION_AUDIT.json", dict(time_utc=now(), status="pass_frozen_heads_and_independent_predictions", rows=receipts,
          new_training=0, no_test_or_clinical_predictions=True,
          legacy_word_char_training_receipt_count_is_shadowed_metadata=True,
          legacy_training_actual_count_by_source_and_manifest=33395))
    files += [OUT / "PREDICTION_AUDIT.json", OUT / "ENVIRONMENT.json"]
    write(OUT / "PREDICTION_SEAL.json", dict(time_utc=now(), files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))


def diagnose(mu, labels):
    p = expit(mu); residual = labels - p; d = p.shape[1]
    diagonal = np.mean(p * (1 - p), axis=0); assert np.all(diagonal > 0)
    D = np.diag(diagonal); M = residual.T @ residual / len(p)
    inverse = 1 / np.sqrt(diagonal); R = inverse[:, None] * M * inverse[None, :]
    eig, vectors = np.linalg.eigh(R); v = vectors[:, -1]
    if v[np.argmax(abs(v))] < 0: v = -v
    direction = inverse * v; direction /= np.linalg.norm(direction); S = np.outer(direction, direction)
    lam = float(eig[-1]); critical = lam / (lam - 1) if lam > 1 else None
    loop_d = np.zeros(d); loop_M = np.zeros((d, d))
    for prob, target in zip(p, labels):
        loop_d += prob * (1 - prob); error = target - prob; loop_M += np.outer(error, error)
    loop_d /= len(p); loop_M /= len(p)
    loop_R = np.empty((d, d))
    for i in range(d):
        for j in range(d): loop_R[i, j] = loop_M[i, j] / np.sqrt(loop_d[i] * loop_d[j])
    assert np.allclose(loop_M, M, atol=1e-11, rtol=1e-11) and np.allclose(loop_R, R, atol=1e-11, rtol=1e-11)
    assert np.max(abs(loop_d - diagonal)) < 1e-11
    checks = dict(D_loop_max_abs=float(np.max(abs(loop_d - diagonal))), M_loop_max_abs=float(np.max(abs(loop_M - M))),
                  R_loop_max_abs=float(np.max(abs(loop_R - R))), eigen_residual=float(np.max(abs(R @ v - lam * v))),
                  S_trace_error=float(abs(np.trace(S) - 1)), S_min_eigenvalue=float(np.linalg.eigvalsh(S)[0]))
    assert checks["eigen_residual"] < 1e-10 and checks["S_trace_error"] < 1e-12 and checks["S_min_eigenvalue"] > -1e-12
    derivatives = []
    for K in CONFIG["K"]:
        coefficient = 1 - 1 / K
        answer = float(.5 * np.trace(S @ (D - coefficient * M)))
        cross = sum(1 for i in range(K) for j in range(K) if i != j) / K**2
        curvature = float(np.mean(np.sum(p * (1 - p) * direction**2, axis=1)))
        gradient_square = float(np.mean((residual @ direction)**2))
        independent = .5 * (curvature - cross * gradient_square)
        spectral = .5 * curvature * (1 - coefficient * lam)
        error = max(abs(answer - independent), abs(answer - spectral))
        assert error < CONFIG["derivative_max_error"] and abs(cross - coefficient) < 1e-15
        assert (answer < 0) == (coefficient * lam > 1)
        derivatives.append(dict(K=K, derivative=answer, projection_derivative=independent, spectral_derivative=spectral,
                                absolute_audit_error=error, PSD_descent_exists=bool(coefficient * lam > 1)))
    checks["derivative_max_error"] = max(x["absolute_audit_error"] for x in derivatives)
    return dict(D=D.tolist(), M=M.tolist(), R=R.tolist(), eigenvalues_ascending=eig.tolist(), top_eigenvector=v.tolist(),
                normalized_direction=direction.tolist(), S=S.tolist(), lambda_max=lam, Kcrit=critical,
                infinite_K_derivative=float(.5 * np.trace(S @ (D - M))),
                marginal_NLL=float(np.mean(np.logaddexp(0, mu) - labels * mu)), marginal_Brier=float(np.mean(residual**2)),
                derivatives=derivatives, audit=checks)


def spectrum():
    check_freeze(); check_seal("DATA_SEAL.json"); check_seal("PREDICTION_SEAL.json")
    assert not (OUT / "COMPLETE.json").exists()
    summaries = []; derivatives = []; audits = []
    for kind in CONFIG["models"]:
        for role in ROLES:
            with np.load(OUT / "cache" / f"{kind}_{role}_means.npz") as z: mu = z["means"]; ids = z["ids"]
            with np.load(OUT / "cache" / (role + ".npz")) as z:
                assert np.array_equal(ids, z["id"]); labels = z["labels"]
            for scale in CONFIG["scales"]:
                for rater in CONFIG["raters"]:
                    key = dict(kind=kind, role=role, scale=scale, rater=rater, primary=rater == 0, n=len(mu))
                    result = diagnose(scale * mu, labels[:, rater])
                    write(OUT / "matrices" / f"{kind}_{role}_scale{scale:g}_rater{rater}.json", dict(**key, **result))
                    summaries.append(dict(**key, **{k: result[k] for k in ["lambda_max", "Kcrit", "infinite_K_derivative", "marginal_NLL", "marginal_Brier"]}))
                    derivatives.extend(dict(**key, **d) for d in result["derivatives"])
                    audits.append(dict(**key, **result["audit"]))
    assert len(summaries) == 36 and len(derivatives) == 216
    csvwrite(OUT / "SUMMARY.csv", summaries); csvwrite(OUT / "DERIVATIVES.csv", derivatives); csvwrite(OUT / "AUDIT_DETAILS.csv", audits)
    write(OUT / "AUDIT.json", dict(time_utc=now(), status="pass_GoEmotions_frozen_empirical_boundary_diagnostic",
          combinations=36, derivative_rows=216, primary_combinations=12, sensitivity_combinations=24,
          max_errors={key: max(a[key] for a in audits) for key in ["D_loop_max_abs", "M_loop_max_abs", "R_loop_max_abs", "eigen_residual", "S_trace_error", "derivative_max_error"]},
          new_mean_fits=0, new_covariance_fits=0, test_evaluation=False, clinical_data=False,
          population_inference=False, global_optimality_certificate=False))
    public = sorted(p for p in OUT.rglob("*") if p.is_file() and p.relative_to(OUT).parts[0] not in ["raw", "cache", "env"])
    write(OUT / "COMPLETE.json", dict(time_utc=now(), status="complete_GDATA", files={p.relative_to(ROOT).as_posix(): sha(p) for p in public}))
    print(read(OUT / "AUDIT.json"), flush=True)


def check():
    check_freeze(); check_seal("DATA_SEAL.json"); check_seal("PREDICTION_SEAL.json"); check_seal("COMPLETE.json")
    assert read(OUT / "AUDIT.json")["status"].startswith("pass_")
    print("pass_GDATA_frozen_sources_local_evidence_and_public_outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["freeze", "environment", "download", "package", "predict", "spectrum", "check"])
    args = parser.parse_args(); globals()[args.phase]()
