"""Descriptive frozen-N trajectories and post-hoc initial-mode projections."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/dependence_boundary_v1"
N = BASE / "N"
SPECTRUM = BASE / "SPECTRUM"
OUT = BASE / "N_ANALYSIS"
DOC = ROOT / "docs/08_research_研究工作/边际依赖接口_20260929/NATURAL_DYNAMICS.md"
FIG = BASE / "figures/natural_dynamics"
PLAN = dict(
    scope="Post-hoc descriptive analysis of all 96 frozen N fits; no training, E scores, or C/T labels.",
    projection="R=D^-1/2 M D^-1/2 at original fixed means; w_k=||L^T D^1/2 v_k||^2.",
    fractions="Report iid-negative(lambda>K/(K-1)) and antithetic-negative(lambda>1) mode fractions for every model, plus own-sampler fraction. Fractions use weighted covariance trace, not explained causal shares.",
    retention="All 16 configurations, 96 replicate endpoints, 48 paired rows, 864 trajectory points, and 2304 modes.",
    uncertainty="Means/min/max across six existing paired models are descriptive; no new confidence intervals or inferential independence claim.",
    local_boundary="Initial real-valued expected-MC curvature only, not final Hessian, global optimum, or optimizer convergence. Source float32 saturation and softplus boundary retained.",
    audit="Frozen input hashes, factor vs direct-quadratic projections, sum-of-mode trace identity, spectral contraction vs trace(Sigma*A), endpoint float32 trace discrepancy, complete rows.",
)

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text.rstrip() + "\n")

def csvwrite(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def verify_manifest(path):
    r = read(path)
    for name, digest in r["files"].items():
        assert sha(ROOT / name) == digest, name

def input_paths():
    return [
        N / "FREEZE.json", N / "MODELS_FREEZE.json", N / "MODEL_AUDIT.csv",
        N / "DATA_RECEIPT.json", N / "AUDIT.json", SPECTRUM / "COMPLETE.json",
        *sorted((N / "models").glob("*.json")),
        *sorted((N / "initial").glob("*.json")),
        *[SPECTRUM / "matrices" / f"old_{r}_{role}_scale1.json"
          for role in ["I", "O"] for r in range(6)],
    ]

def build():
    endpoints, trajectories, modes = [], [], []
    errors = dict(projection=0., mode_sum=0., curvature=0., trace_float32=0.)
    for path in sorted((N / "models").glob("*.json")):
        model = read(path)
        config, ident = model["config"], model["id"]
        initial = read(N / "initial" / f"{ident}.json")
        K = config["K"]
        assert len(model["trace"]) == 8
        for rep in range(6):
            L = np.asarray(model["factors"][rep])
            cov = L @ L.T
            np.testing.assert_allclose(cov, model["covariance"][rep], rtol=1e-13, atol=1e-13)
            sp = read(SPECTRUM / "matrices" / f'old_{rep}_{config["role"]}_scale1.json')
            D, M, R = (np.asarray(sp[k]) for k in ["D", "M", "R"])
            rootD = np.sqrt(np.diag(D))
            np.testing.assert_allclose(R, M / rootD[:, None] / rootD[None, :], rtol=1e-12, atol=1e-12)
            eigen, V = np.linalg.eigh(R)
            np.testing.assert_allclose(eigen, sp["eigenvalues_ascending"], rtol=1e-12, atol=1e-12)
            projected_factor = L.T @ (rootD[:, None] * V)
            energy = (projected_factor * projected_factor).sum(0)
            weighted_cov = rootD[:, None] * cov * rootD[None, :]
            quadratic = np.diag(V.T @ weighted_cov @ V)
            total = float(np.trace(weighted_cov))
            assert total > 0
            errors["projection"] = max(errors["projection"], float(np.max(np.abs(energy - quadratic))))
            errors["mode_sum"] = max(errors["mode_sum"], abs(float(energy.sum()) - total))
            masks = dict(iid=eigen > K / (K - 1), antithetic=eigen > 1.)
            slopes = {}
            for sampler in ["iid", "antithetic"]:
                coefficient = 1 - (1 - 1 / K) * eigen if sampler == "iid" else 1 - eigen
                A = D - (1 - 1 / K) * M if sampler == "iid" else D - M
                slopes[sampler] = float(.5 * energy @ coefficient)
                direct = float(.5 * np.trace(cov @ A))
                errors["curvature"] = max(errors["curvature"], abs(slopes[sampler] - direct))
            initial_cov = np.asarray(initial["covariance"][rep])
            initial_trace, final_trace = float(np.trace(initial_cov)), float(np.trace(cov))
            fractions = {s: float(energy[m].sum() / total) for s, m in masks.items()}
            final_record = model["trace"][-1]
            errors["trace_float32"] = max(errors["trace_float32"],
                abs(final_trace - final_record["post_update_tracevariance"][rep]))
            assert np.isclose(final_trace, final_record["post_update_tracevariance"][rep], rtol=1e-6, atol=1e-8)
            row = dict(
                id=ident, **config, rep=rep, seed=model["seeds"][rep],
                lambda_max=float(eigen[-1]), iid_threshold=K / (K - 1),
                iid_negative_mode_count=int(masks["iid"].sum()),
                anti_negative_mode_count=int(masks["antithetic"].sum()),
                iid_descent_exists=bool(masks["iid"].any()),
                anti_descent_exists=bool(masks["antithetic"].any()),
                initial_tracevariance=initial_trace, final_tracevariance=final_trace,
                trace_change=final_trace - initial_trace, trace_ratio=final_trace / initial_trace,
                weighted_covariance_trace=total,
                energy_fraction_iid_negative=fractions["iid"],
                energy_fraction_anti_negative=fractions["antithetic"],
                energy_fraction_own_negative=fractions[config["sampler"]],
                zero_boundary_slope_in_final_cov_direction_iid=slopes["iid"],
                zero_boundary_slope_in_final_cov_direction_anti=slopes["antithetic"],
                final_minibatch_concept_NLL=final_record["pre_update_concept_NLL"][rep],
                minimum_numerical_eigenvalue=float(np.linalg.eigvalsh(cov)[0]),
                factor_diag_min=float(np.diag(L).min()),
            )
            endpoints.append(row)
            trajectories.append(dict(
                id=ident, **config, rep=rep, step=0,
                tracevariance=initial_trace, minibatch_concept_NLL=None,
                sigmoid_zero=None, sigmoid_one=None, probability_values=None))
            for rec in model["trace"]:
                trajectories.append(dict(
                    id=ident, **config, rep=rep, step=rec["total_step"],
                    tracevariance=rec["post_update_tracevariance"][rep],
                    minibatch_concept_NLL=rec["pre_update_concept_NLL"][rep],
                    sigmoid_zero=rec["pre_update_sigmoid_zero"][rep],
                    sigmoid_one=rec["pre_update_sigmoid_one"][rep],
                    probability_values=rec["probability_values_per_rep"]))
            for j in range(24):
                modes.append(dict(
                    id=ident, **config, rep=rep, mode=j, eigenvalue=float(eigen[j]),
                    projected_energy=float(energy[j]), energy_fraction=float(energy[j] / total),
                    iid_negative=bool(masks["iid"][j]), anti_negative=bool(masks["antithetic"][j])))
    assert (len(endpoints), len(trajectories), len(modes)) == (96, 864, 2304)
    assert max(errors[k] for k in ["projection", "mode_sum", "curvature"]) < 1e-11
    return endpoints, trajectories, modes, errors

def summaries(endpoints):
    groups, pairs = [], []
    for role in ["I", "O"]:
        for sd in [.01, 1.]:
            for K in [4, 100]:
                for sampler in ["iid", "antithetic"]:
                    rr = [r for r in endpoints if r["role"] == role and r["initial_sd"] == sd
                          and r["K"] == K and r["sampler"] == sampler]
                    assert len(rr) == 6
                    v = np.asarray([r["final_tracevariance"] for r in rr])
                    groups.append(dict(
                        role=role, initial_sd=sd, K=K, sampler=sampler,
                        initial_tracevariance=rr[0]["initial_tracevariance"],
                        final_trace_mean=float(v.mean()), final_trace_min=float(v.min()),
                        final_trace_max=float(v.max()),
                        initial_iid_descent_models=sum(r["iid_descent_exists"] for r in rr),
                        initial_anti_descent_models=sum(r["anti_descent_exists"] for r in rr),
                        mean_energy_fraction_iid_negative=float(np.mean([r["energy_fraction_iid_negative"] for r in rr])),
                        mean_energy_fraction_anti_negative=float(np.mean([r["energy_fraction_anti_negative"] for r in rr])),
                        mean_final_minibatch_NLL=float(np.mean([r["final_minibatch_concept_NLL"] for r in rr]))))
                for rep in range(6):
                    rr = {r["sampler"]: r for r in endpoints if r["role"] == role
                          and r["initial_sd"] == sd and r["K"] == K and r["rep"] == rep}
                    a, b = rr["iid"], rr["antithetic"]
                    pairs.append(dict(
                        role=role, initial_sd=sd, K=K, rep=rep, lambda_max=a["lambda_max"],
                        iid_descent_exists=a["iid_descent_exists"], anti_descent_exists=a["anti_descent_exists"],
                        iid_final_trace=a["final_tracevariance"], anti_final_trace=b["final_tracevariance"],
                        anti_minus_iid_trace=b["final_tracevariance"] - a["final_tracevariance"],
                        anti_over_iid_trace=b["final_tracevariance"] / a["final_tracevariance"],
                        iid_energy_in_iid_negative_modes=a["energy_fraction_iid_negative"],
                        anti_energy_in_iid_negative_modes=b["energy_fraction_iid_negative"],
                        iid_energy_in_anti_negative_modes=a["energy_fraction_anti_negative"],
                        anti_energy_in_anti_negative_modes=b["energy_fraction_anti_negative"]))
    assert len(groups) == 16 and len(pairs) == 48
    return groups, pairs

def plot(trajectories):
    assert not FIG.with_suffix(".png").exists() and not FIG.with_suffix(".svg").exists()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), sharex=True)
    colors = {(4, "iid"): "#2F5597", (4, "antithetic"): "#E08214",
              (100, "iid"): "#128C83", (100, "antithetic"): "#9C4587"}
    steps = np.asarray([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    for i, sd in enumerate([.01, 1.]):
        for j, role in enumerate(["I", "O"]):
            ax = axes[i, j]
            for K, sampler in colors:
                values = []
                for rep in range(6):
                    rr = sorted([r for r in trajectories if r["role"] == role and r["initial_sd"] == sd
                        and r["K"] == K and r["sampler"] == sampler and r["rep"] == rep], key=lambda x: x["step"])
                    assert [r["step"] for r in rr] == steps.tolist()
                    values.append([r["tracevariance"] for r in rr])
                values = np.asarray(values)
                label = f"K={K}, " + ("iid" if sampler == "iid" else "antithetic")
                ax.plot(steps, values.mean(0), label=label, color=colors[(K, sampler)],
                        linestyle="--" if sampler == "iid" else "-", linewidth=2)
                ax.fill_between(steps, values.min(0), values.max(0),
                                color=colors[(K, sampler)], alpha=.09, linewidth=0)
            ax.set_yscale("log")
            ax.axvline(1500, color="#777777", lw=.8, ls=":")
            ax.grid(alpha=.16, which="both")
            ax.set_title(f"Role {role} | initial latent SD = {sd:g}", fontsize=12)
            if j == 0:
                ax.set_ylabel("Covariance trace (log scale)")
            if i == 1:
                ax.set_xlabel("Optimization step")
            ax.set_xticks([0, 500, 1000, 1500, 2000])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .042), frameon=False)
    fig.suptitle("Sampling design changes covariance growth", fontsize=16, y=.98)
    fig.text(.5, .925, "All 96 frozen fits; line = six-model mean, band = min-max", ha="center", color="#444444")
    fig.text(.5, .012, "Dotted vertical line: Adam/RNG reset. Initial curvature is not a trajectory or risk guarantee.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[.02, .10, .99, .90])
    fig.savefig(FIG.with_suffix(".png"), dpi=180)
    fig.savefig(FIG.with_suffix(".svg"))
    plt.close(fig)
    p = FIG.with_suffix(".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text(encoding="utf-8").splitlines()) + "\n",
                 encoding="utf-8", newline="\n")

def documents(groups, pairs, endpoints, errors):
    table = ["|角色|初始SD|K|采样|末方差迹均值|六模型范围|起点iid下降/6|",
             "|---|---:|---:|---|---:|---:|---:|"]
    for r in groups:
        table.append(f'|{r["role"]}|{r["initial_sd"]:g}|{r["K"]}|{r["sampler"]}|'
                     f'{r["final_trace_mean"]:.6f}|[{r["final_trace_min"]:.6f}, {r["final_trace_max"]:.6f}]|{r["initial_iid_descent_models"]}|')
    rep = ["# 全部配对重复：初始判据与实际响应", "",
           "描述性逐例表；不把一个模型内的模态、步数或两个初始化作为独立科学重复。", "",
           "|角色|SD|K|rep|lambda max|iid有下降|anti有下降|iid末迹|anti末迹|anti−iid|",
           "|---|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for r in pairs:
        rep.append(f'|{r["role"]}|{r["initial_sd"]:g}|{r["K"]}|{r["rep"]}|{r["lambda_max"]:.6f}|'
                   f'{r["iid_descent_exists"]}|{r["anti_descent_exists"]}|{r["iid_final_trace"]:.6f}|'
                   f'{r["anti_final_trace"]:.6f}|{r["anti_minus_iid_trace"]:.6f}|')
    write_text(OUT / "REPLICATES_READABLE.md", "\n".join(rep))
    full = "\n".join(table)
    # Shared initial mode sets make the two samplers directly comparable.
    spectral = ["|角色|SD|K|iid落入iid负模态|anti落入同一iid负模态|iid落入anti负模态|anti落入同一anti负模态|",
                "|---|---:|---:|---:|---:|---:|---:|"]
    for role in ["I", "O"]:
        for sd in [.01, 1.]:
            for K in [4, 100]:
                rr = [r for r in pairs if r["role"] == role and r["initial_sd"] == sd and r["K"] == K]
                vals = [np.mean([r[k] for r in rr]) for k in [
                    "iid_energy_in_iid_negative_modes", "anti_energy_in_iid_negative_modes",
                    "iid_energy_in_anti_negative_modes", "anti_energy_in_anti_negative_modes"]]
                spectral.append(f"|{role}|{sd:g}|{K}|" + "|".join(f"{v:.6f}" for v in vals) + "|")
    source_audit = read(N / "AUDIT.json")
    document = f"""# 采样设计改变了协方差生长，但局部判据不决定整条轨迹

本分析只读取已经封存的N训练与原I/O谱文件，不读取E测试分数、C/T标签或新均值数据。全部96拟合、两个初始化和所有重复保留。以下是事后描述性动力学；六模型的均值/范围不构成新的置信区间。

## 初始判据与观察到的响应

旧MLP/I、K4的六个模型中，**5/6在零协方差处没有iid负曲率方向，1/6有；成对采样6/6有**。这个判据对两种初始化使用同一零点，不能说sd=1初始点也满足该Hessian结论。

sd=.01时，I/K4的iid末方差迹均值为.045928，成对采样为1.817515；sd=1时分别为3.092947、4.434426。前一组呈现较强的协方差生长差异，后一组也保留差异，幅度不同。**iid小初始化的迹仍从约.0024上升到.0459**，因此数据不支持“没有人口下降方向就逐步保持零方差”的机械叙事。

K100/I的差异小得多：sd=.01时4.305914与4.690994，sd=1时7.857762与8.001684。O角色下，两种采样器在两种K和初始化下都形成大得多的方差迹；K100/sd=.01的anti迹略低于iid，完整保留。方差更大本身不是风险更低或依赖更正确的证据。

{full}

![全部配置的方差迹](../../../artifacts/dependence_boundary_v1/figures/natural_dynamics.png)

每条曲线为六模型均值，阴影为最小—最大；虚线竖标记1500步后Adam和随机流重置。日志方差迹在更新后记录；同条日志的概念NLL来自更新前当前minibatch，并已补logK。它不是整训练集精确目标或测试风险，不能直接用其跨K/采样器排名替代独立评价。

## 投影到初始模态的事后诊断

对每个原I/O模型，复用初始经验残差矩阵M及D，令R=D^(-1/2) M D^(-1/2)，特征向量为v。最终协方差的模态能量定义为vᵀD^(1/2)ΣD^(1/2)v，计算时采用等价的范数平方以避免近奇异矩阵产生舍入负能量。

下表对两个采样器分别使用**同一初始模态集合**：iid负模态为lambda>K/(K−1)，anti负模态为lambda>1。分母是最终加权协方差迹tr(DΣ)。若一个模型没有指定负模态，比例严格记0；没有把没有模态的模型剔除。

{chr(10).join(spectral)}

这些比例只是固定基底中的加权方差分配，不是因果份额、信息比例、已解释性能或最终最优谱阈值。两个阈值集合不同，所以不能直接比较各自“own negative”比例来宣称哪个方法更好。逐模态、逐模型和共同集合的配对数据全部保留。

另外保存.5 tr(Σ_final A_initial)与模态加权和的等式核验。它表示从零点沿最终协方差方向出发的起始斜率，**不是最终点的梯度或曲率**。协方差变大后，旧起点矩阵不能代替重算当前目标。

## 执行与数值边界

source采用softplus(rawdiag)+1e−6，零factor只在闭包中；sd=.01也不是精确零点。当前运行使用带minibatch及MC噪声的Adam，两采样器既改变目标也改变噪声；有限步响应不单独识别这两种机制的因果贡献。训练外角色、上游均值泛化和数据标签结构同样参与其经验残差谱。

原均值及全部日志的float32 sigmoid饱和计数随N保留。理论谱使用实数logistic目标，不能把它当作含饱和BCE源码的逐位梯度证书。96模型均由严格正对角三角factor给出结构正定协方差；14个sd=.01模型的数值最小特征值非正，最负约−7.96e−16，属于已冻结舍入容差，不是真实负方差。sd=1全部数值正定，最小特征值约.02978。

12个K100/iid/sd1终点与旧R/N1的raw/factor逐值相同，是本轮默认锚重放，不能计作12个新的独立科学发现。全部96次训练执行、84个非默认锚终点及其数据复用关系保持。

## 只读核验和产物

冻结输入、96末端、864轨迹点及2304模态均核对；因子范数投影与直接二次型最大差{errors["projection"]:.3g}，模态和与加权迹最大差{errors["mode_sum"]:.3g}，谱斜率与矩阵迹最大差{errors["curvature"]:.3g}。float32日志末迹与float64因子重构最大差{errors["trace_float32"]:.3g}，满足预定相对容差1e−6。

[逐重复可读表](../../../artifacts/dependence_boundary_v1/N_ANALYSIS/REPLICATES_READABLE.md) · [完整末端](../../../artifacts/dependence_boundary_v1/N_ANALYSIS/ENDPOINTS.csv) · [逐模态](../../../artifacts/dependence_boundary_v1/N_ANALYSIS/MODES.csv) · [分析审计](../../../artifacts/dependence_boundary_v1/N_ANALYSIS/AUDIT.json)。

只读入口：.venv/Scripts/python -m script.dependence_boundary_dynamics --check。没有读取测试分数、添加训练或覆盖N的冻结。
"""
    write_text(DOC, document)

def run():
    assert not OUT.exists(), "Refuse to overwrite analysis outputs"
    verify_manifest(N / "FREEZE.json")
    verify_manifest(N / "MODELS_FREEZE.json")
    verify_manifest(SPECTRUM / "COMPLETE.json")
    write(OUT / "INPUTS.json", dict(time_utc=now(), kind="posthoc_descriptive",
        plan=PLAN, files={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), *input_paths()]}))
    endpoints, trajectories, modes, errors = build()
    groups, pairs = summaries(endpoints)
    gate = [r for r in endpoints if r["role"] == "I" and r["K"] == 4
            and r["initial_sd"] == .01 and r["sampler"] == "iid"]
    assert len(gate) == 6 and sum(r["iid_descent_exists"] for r in gate) == 1
    assert sum(r["anti_descent_exists"] for r in gate) == 6
    csvwrite(OUT / "ENDPOINTS.csv", endpoints)
    csvwrite(OUT / "TRAJECTORIES.csv", trajectories)
    csvwrite(OUT / "MODES.csv", modes)
    csvwrite(OUT / "SUMMARY.csv", groups)
    csvwrite(OUT / "PAIRED.csv", pairs)
    plot(trajectories)
    documents(groups, pairs, endpoints, errors)
    write(OUT / "AUDIT.json", dict(time_utc=now(), passed=True, errors=errors,
        configurations=16, endpoints=96, paired_rows=48, trajectory_points=864,
        modes=2304, test_scores_or_labels_read=False, additional_training=0))
    paths = [*sorted(OUT.glob("*")), DOC, FIG.with_suffix(".png"), FIG.with_suffix(".svg")]
    write(OUT / "MANIFEST.json", dict(time_utc=now(),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}))
    check()

def check():
    verify_manifest(N / "FREEZE.json")
    verify_manifest(N / "MODELS_FREEZE.json")
    verify_manifest(SPECTRUM / "COMPLETE.json")
    verify_manifest(OUT / "INPUTS.json")
    verify_manifest(OUT / "MANIFEST.json")
    endpoints, trajectories, modes, errors = build()
    for filename, expected in [
        ("ENDPOINTS.csv", endpoints), ("TRAJECTORIES.csv", trajectories), ("MODES.csv", modes)]:
        with (OUT / filename).open(encoding="utf-8") as stream:
            actual = list(csv.DictReader(stream))
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            for key, val in b.items():
                assert a[key] == ("" if val is None else str(val)), (filename, key)
    print(json.dumps(dict(pass_analysis=True, endpoints=96, modes=2304, errors=errors)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else run()
