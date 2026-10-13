"""Independent bounded-scope audit of LOCAL; never imports its producer.

Freeze this source and all evidence before replay. RNG is replay only, never a
new scientific pool. The probability-domain Price reconstruction covers the
first 64 banks of each stream; all saved statistical arithmetic is rebuilt.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
PRODUCER = ROOT / "script/bankvar_local.py"
BASE = ROOT / "artifacts/bank_precision/LOCAL"
OUT = ROOT / "artifacts/bank_precision/LOCAL_AUDIT"
PROTOCOL = OUT / "PROTOCOL.json"
DOC = ROOT / "docs/08_research_研究工作/粒子精度与局部校准_20261002/LOCAL_AUDIT.md"
RHO = np.array([-.12, -.06, -.03, .03, .06, .12])
H = np.array([.03, .06, .12])
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")


def freeze():
    assert not PROTOCOL.exists(), "Audit already frozen"
    p = read(BASE / "PROTOCOL.json")
    assert (BASE / "MODEL.json").is_file(), "Producer must have completed"
    paths = [PRODUCER, BASE / "PROTOCOL.json", BASE / "EXECUTION.json", BASE / "MODEL.json"]
    paths += [ROOT / name for name in p["inputs"]]
    paths += [BASE / f"stream_{i:02}.{ext}" for i in range(8) for ext in ("json", "npz")]
    inputs = {rel(path): sha(path) for path in paths}
    write_new(PROTOCOL, dict(schema="bankvar-local-independent-audit-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(SOURCE),
        inputs=inputs, rng_replay="8 streams, each 16 sequential PCG64 normal blocks (64,256,256)",
        price_scope="First64 banks of every stream; all6 rho and4 active modes; direct sigmoid probability product and normalization",
        saved_arithmetic_scope="Every G, c_h, stream mean, pooled coefficient, across8stream SE, paired resolution SE, both gates and four2048bank subpools",
        probability_price_tolerance=5e-13, arithmetic_tolerance=1e-12,
        new_independent_scientific_samples=0, automatic_retry=False))
    print(json.dumps(dict(frozen=True, source_sha256=sha(SOURCE), protocol_sha256=sha(PROTOCOL))), flush=True)


class Checks:
    def __init__(self):
        self.rows = []

    def flag(self, name, condition):
        value = bool(condition)
        self.rows.append(dict(name=name, passed=value))
        return value

    def close(self, name, actual, expected, tolerance):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        err = float(np.max(np.abs(a-b))) if valid and a.size else (0. if valid else 1e300)
        self.rows.append(dict(name=name, passed=bool(valid and err <= tolerance),
                             shape=list(a.shape), max_absolute_error=err, tolerance=tolerance))
        return err


def independent_price(z):
    """Direct positive probabilities, no production logsumexp or imports."""
    noise = np.prod(expit(z[:, :, 2:]), axis=2)
    assert np.all(noise > 0) and np.isfinite(noise).all()
    ans = np.empty((len(z), 6, 4))
    for ri, rho in enumerate(RHO):
        x2 = rho*z[:, :, 0]+math.sqrt(1-rho*rho)*z[:, :, 1]
        for si, (s1, s2) in enumerate(SIGNS):
            first, second = expit(s1*z[:, :, 0]), expit(s2*x2)
            likelihood = noise*first*second
            weights = likelihood/np.sum(likelihood, axis=1)[:, None]
            comp = (1-first)*(1-second)
            first_moment = np.sum(weights*comp, axis=1)
            squared_weight_moment = np.sum(weights*weights*comp, axis=1)
            ans[:, ri, si] = -int(s1*s2)*(first_moment-squared_weight_moment)
    assert np.isfinite(ans).all()
    return ans


def mean_columns(x):
    return np.array([math.fsum(float(v) for v in x[:, j])/len(x) for j in range(x.shape[1])])


def independent_se(x):
    center = mean_columns(x)
    m = len(x)
    return np.array([math.sqrt(math.fsum((float(v)-center[j])**2 for v in x[:, j])/(m*(m-1)))
                     for j in range(x.shape[1])])


def run_audit():
    frozen = read(PROTOCOL)
    assert frozen["source_sha256"] == sha(SOURCE), "Audit source changed after freeze"
    for name, digest in frozen["inputs"].items():
        assert sha(ROOT/name) == digest, f"Audit input changed: {name}"
    assert not (OUT / "EXECUTION.json").exists(), "No automatic replay after prior attempt"
    write_new(OUT / "EXECUTION.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE), protocol_sha256=sha(PROTOCOL), numpy_version=np.__version__,
        scope="Authorized replay, not independent scientific draws"))
    start = time.perf_counter()
    checks, stream_results = Checks(), []
    try:
        p, model, execution = read(BASE/"PROTOCOL.json"), read(BASE/"MODEL.json"), read(BASE/"EXECUTION.json")
        reference_path = ROOT/"artifacts/finite_calibration/CAL/REFERENCE.json"
        reference = read(reference_path)
        checks.flag("producer/source_binding", p["source_sha256"] == sha(PRODUCER))
        for name, digest in p["inputs"].items():
            checks.flag("producer/input/"+name, sha(ROOT/name) == digest)
        for name, obj in (("model", model), ("execution", execution)):
            checks.flag(name+"/source", obj["source_sha256"] == sha(PRODUCER))
            checks.flag(name+"/protocol", obj["protocol_sha256"] == sha(BASE/"PROTOCOL.json"))
            checks.flag(name+"/reference", obj["reference_sha256"] == sha(reference_path))
        checks.flag("protocol/fixed_design", p["a"] == 1 and p["R"] == .75 and p["n"] == 254 and p["K"] == 256
                    and p["streams"] == 8 and p["banks_per_stream"] == 1024 and p["block"] == 64
                    and p["seed_base"] == 202610021700 and p["h_main"] == .06)
        checks.close("protocol/rho", p["rho"], RHO, 0.)
        checks.close("protocol/h", p["h"], H, 0.)
        checks.close("protocol/signs", p["active_signs"], SIGNS, 0.)
        grid = np.asarray(reference["rho801"])
        table = next(row for row in reference["orders"] if row["order"] == reference["main_order"])
        indices = [int(np.argmin(abs(grid-v))) for v in RHO]
        rr = np.asarray(table["r801"])[indices]
        checks.close("reference/node_match", grid[indices], RHO, 1e-14)
        checks.close("reference/indices", p["reference_indices"], indices, 0.)
        checks.close("reference/r", p["reference_r"], rr, 0.)
        checks.flag("reference/qualification", reference["reference_qualification_pass"] and reference["r_qualification_pass"])
        # Static capability check supplements full manual source review. The
        # producer has no imports of any old producer/helper or data connector.
        tree = ast.parse(PRODUCER.read_text("utf-8"))
        imports = sorted({node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} |
                         {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names})
        checks.flag("source/no_old_producer_import", not any("fincal" in x or "reglearn" in x for x in imports))
        code = PRODUCER.read_text("utf-8")
        checks.flag("source/only_known_reference_path", 'artifacts/finite_calibration/CAL/REFERENCE.json' in code
                    and 'artifacts/finite_calibration/CAL/MODEL.json' not in code
                    and 'artifacts/finite_calibration/TRAIN' not in code)
        stream_means = []
        for stream in range(8):
            meta = read(BASE/f"stream_{stream:02}.json")
            file = BASE/f"stream_{stream:02}.npz"
            checks.flag(f"stream{stream}/metadata", meta["stream"] == stream and meta["seed"] == 202610021700+stream
                        and meta["banks"] == 1024 and meta["file"] == rel(file) and meta["sha256"] == sha(file))
            checks.flag(f"stream{stream}/model_copy", model["streams"][stream] == meta)
            with np.load(file, allow_pickle=False) as archive:
                D = archive["D_modes"].copy()
                saved_G, saved_c = archive["G_bank"].copy(), archive["coefficients"].copy()
                for name, expected in (("rho", RHO), ("h", H), ("signs", SIGNS), ("r", rr)):
                    checks.close(f"stream{stream}/array_{name}", archive[name], expected, 0.)
            checks.flag(f"stream{stream}/D_shape_finite", D.shape == (1024, 6, 4) and np.isfinite(D).all())
            checks.flag(f"stream{stream}/Price_sign", np.all(D*np.prod(SIGNS, axis=1) <= 0.))
            rng = np.random.Generator(np.random.PCG64(202610021700+stream))
            hasher, first_digest, price_error = hashlib.sha256(), None, None
            for offset in range(0, 1024, 64):
                z = rng.standard_normal((64, 256, 256))
                raw = z.tobytes()
                hasher.update(raw)
                if offset == 0:
                    first_digest = hashlib.sha256(raw).hexdigest()
                    direct = independent_price(z)
                    price_error = checks.close(f"stream{stream}/independent_probability_Price_first64", direct, D[:64],
                                               frozen["probability_price_tolerance"])
            checks.flag(f"stream{stream}/entire_prior_rng", hasher.hexdigest() == meta["prior_sha256"])
            checks.flag(f"stream{stream}/first_block_rng", first_digest == meta["first_block_prior_sha256"])
            # U and V projection is algebraically distinct from multiplying
            # each saved mode by (1+r*s1*s2) before summation.
            U = (D[:, :, 0]+D[:, :, 1]+D[:, :, 2]+D[:, :, 3])/4
            V = (D[:, :, 0]-D[:, :, 1]-D[:, :, 2]+D[:, :, 3])/4
            G = -U-V*rr[None, :]
            C = np.column_stack([(G[:, 3]-G[:, 2])/.12,
                                 (G[:, 4]-G[:, 1])/.24,
                                 (G[:, 5]-G[:, 0])/.48])
            checks.close(f"stream{stream}/all_G", G, saved_G, frozen["arithmetic_tolerance"])
            checks.close(f"stream{stream}/all_coefficients", C, saved_c, frozen["arithmetic_tolerance"])
            mc = mean_columns(C)
            checks.close(f"stream{stream}/mean_G", mean_columns(G), meta["mean_G"], frozen["arithmetic_tolerance"])
            checks.close(f"stream{stream}/mean_c", mc, meta["mean_c_h"], frozen["arithmetic_tolerance"])
            stream_means.append(mc)
            stream_results.append(dict(stream=stream, replayed_banks=1024, independently_scored_banks=64,
                independent_Price_entries=64*6*4, max_Price_absolute_error=price_error,
                entire_prior_sha256=hasher.hexdigest(), first_prior_sha256=first_digest,
                rebuilt_mean_c_h=mc.tolist()))
            print(json.dumps(dict(audited_stream=stream, Price_max_error=price_error)), flush=True)
        sm = np.array(stream_means)
        pooled, ses = mean_columns(sm), independent_se(sm)
        paired_se = independent_se((sm[:, 1]-sm[:, 0])[:, None])[0]
        for name, actual, expected in (("pooled_c", pooled, model["pooled_c_h"]),
                                      ("stream_SE", ses, model["stream_SE_c_h"]),
                                      ("coefficient", pooled[1], model["coefficient"]),
                                      ("coefficient_SE", ses[1], model["coefficient_SE"]),
                                      ("paired_resolution_SE", paired_se, model["paired_c06_minus_c03_SE"])):
            checks.close(name, actual, expected, frozen["arithmetic_tolerance"])
        checks.flag("fixed_main_coefficient", model["h_main"] == .06 and model["main_coefficient_not_clipped_or_selected"]
                    and model["teacher_specific_tuning"] is False)
        gate_values = dict(main_stream_SE=(float(ses[1]), 2e-4),
                           resolution_c06_c03=(float(abs(pooled[1]-pooled[0])), 5e-5))
        for name, (value, threshold) in gate_values.items():
            gate = model["qualification"][name]
            checks.close("gate/"+name+"/value", value, gate["value"], frozen["arithmetic_tolerance"])
            checks.flag("gate/"+name+"/threshold_pass", gate["threshold"] == threshold and gate["passed"] == (value <= threshold))
        checks.flag("gates/overall", model["qualification_all_passed"] == all(v <= t for v, t in gate_values.values()))
        for i in range(4):
            row = model["subpools_2048"][i]
            checks.flag(f"subpool{i}/definition", row["subpool"] == i and row["streams"] == [2*i, 2*i+1] and row["banks"] == 2048)
            checks.close(f"subpool{i}/all_c", mean_columns(sm[2*i:2*i+2]), row["c_h"], frozen["arithmetic_tolerance"])
        accounting = dict(new_observations=0, new_data_fits=0, new_integrals=0, banks=8192,
                          prior_vectors=2097152, Gaussian_scalars=536870912, bank_rho_mode_evaluations=196608,
                          old_full_CAL_training_banks_for_comparison=32768, bank_reduction_ratio=4.0)
        checks.flag("production/accounting", model["accounting"] == accounting and p["accounting"] == accounting)
        result = dict(schema=1, passed=all(x["passed"] for x in checks.rows),
            source_sha256=sha(SOURCE), protocol_sha256=sha(PROTOCOL), producer_sha256=sha(PRODUCER),
            model_sha256=sha(BASE/"MODEL.json"), checks=checks.rows, stream_results=stream_results,
            producer_imported=False, producer_imports_static=imports, rebuilt_pooled_c_h=pooled.tolist(),
            rebuilt_stream_SE_c_h=ses.tolist(), elapsed_seconds=time.perf_counter()-start,
            audit_accounting=dict(replayed_banks=8192, replayed_prior_vectors=2097152,
                                  replayed_Gaussian_scalars=536870912, independently_Price_scored_banks=512,
                                  independent_Price_entries=12288, saved_Price_entries_checked=196608,
                                  new_independent_scientific_samples=0, new_integrals=0),
            limitation="All prior RNG replayed; direct Price only512 banks. Other modes are checked through all saved arithmetic, not a full independent Price recomputation. Known r reference reused, not a strict integral certificate.")
        write_new(OUT/"RESULT.json", result)
        assert result["passed"], "Audit checks failed; preserve report and review"
        lines = ["# LOCAL 独立实现审计", "", "全部固定审计项通过；不改生产源、协议、数组或门。", "",
                 "8×1024 个 prior bank 的完整 PCG64 序列已重放，全流及首块 SHA256 均与生产记录一致。每流前64 bank、六个 rho、四模式以直接 sigmoid 概率乘积和独立归一化重算 Price，共512 bank/12288项；未导入生产程序或旧 helper。", "",
                 f"独立 Price 最大绝对差：`{max(x['max_Price_absolute_error'] for x in stream_results):.17g}`，固定阈值 `5e-13`。", "",
                 "全部保存的196608个模式项参与独立 U/V 投影、G 与 c_h 重建；所有流均值、pool、8流SE、配对步长SE、两资格门和四个2048bank子池均复核。其它7680 bank的底层Price未独立重算，不能称全生产评分重跑。", "",
                 "生产源静态阅读及AST核验确认：唯一外部科学文件是旧已知发射模型 REFERENCE.json；算法只抽取六个 r801 值，不读取旧拟合 g 或观测文件。完整参考JSON被读入不等于其中所有字段参与校准。没有教师风险调参、步长更换或系数截正。", "",
                 "|h|复建系数|8流SE|", "|---|---:|---:|"]
        lines += [f"|{hh:.2f}|{cc:.15g}|{ss:.15g}|" for hh,cc,ss in zip(H, pooled, ses)]
        lines += ["", "主系数固定 h=.06；本审计通过不把开发SE升级为严格覆盖，也不认证全区间二次修复。", "",
                  "审计重放536870912个Gaussian标量，全部对应已存在科学输入；0新增独立样本、0积分、0训练。", "",
                  f"审计源 SHA256 `{sha(SOURCE)}`；审计协议 `{sha(PROTOCOL)}`；模型 `{sha(BASE/'MODEL.json')}`。", "",
                  "完整逐项收据在 `artifacts/bank_precision/LOCAL_AUDIT/RESULT.json`，源及输入在运行前由 PROTOCOL.json 固定。", ""]
        assert not DOC.exists(), "Refuse audit report overwrite"
        DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(json.dumps(dict(passed=result["passed"], checks=len(checks.rows), elapsed_seconds=result["elapsed_seconds"])), flush=True)
    except Exception as exc:
        write_new(OUT/"FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), checks=checks.rows,
                  completed_streams=stream_results, elapsed_seconds=time.perf_counter()-start))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else run_audit()
