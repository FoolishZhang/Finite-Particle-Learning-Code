"""Display sealed covariance-boundary evidence; no fits, RNG, or quadrature.

Normal-limit curves evaluate erf from previously qualified scalar constants.
The two Monte Carlo contrast families retain their own original intervals.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/covariance_boundary"
OUT = BASE/"figures"
DOC = ROOT/"docs/08_research_研究工作/相关结构补偿与发射噪声_20261001"
NAMES = ("teacher", "fixedRfloor", "comp", "half", "over", "independent")
FIGURES = ("compensation_risk", "score_pair_across_K", "negative_noise_limit")
EXPECTED_AUDITS = {
    "NUMERIC": "2e79c720289d56bafa0add7362bc4b0ecd0d6cb6d79341b65bf4eea798452e58",
    "PROFILE": "7e9878070f637cbc73d45b57f19b170f85b40783512745dfd9ffa5c255d1ced4",
    "HIGHK": "07bd4b2a036f540c77881ccb86364765fc87a10a4c667e82b1e555cd25b6fe96",
}
CHANNEL_SHA = "498b39558140ad7416ddc29a30ed57f68f5129023baf9afc77392af0877c7035"
COLORS = {"teacher": "#222222", "fixedRfloor": "#3d78a6", "comp": "#208c70",
          "half": "#9467a7", "over": "#d07732", "independent": "#b85161"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    value = "\n".join(line.rstrip() for line in value.splitlines()).rstrip()+"\n"
    Path(path).write_text(value, encoding="utf-8", newline="\n")


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def verify(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


def n(value):
    return f"{value:.9g}"


def table(headers, rows):
    return "\n".join(["|"+"|".join(headers)+"|", "|"+"|".join(["---"]*len(headers))+"|"]
                     +["|"+"|".join(str(x) for x in row)+"|" for row in rows])


def csv_file(name, rows):
    assert rows
    with (OUT/name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load():
    sources = [Path(__file__).resolve(), DOC/"CHANNEL_THEORY.md", DOC/"CHANNEL_REVIEW.md",
               DOC/"COMPENSATION_THEORY.md", DOC/"PLAN_NUMERIC.md",
               DOC/"PLAN_PROFILE.md", DOC/"PLAN_HIGHK.md", DOC/"PROFILE_IDENTITY.md"]
    assert sha(DOC/"CHANNEL_THEORY.md") == CHANNEL_SHA
    packages = {}
    for family in ("NUMERIC", "PROFILE", "HIGHK"):
        folder = BASE/family
        assert read(folder/"STATE.json")["status"] == "complete"
        source, result = read(folder/"FREEZE.json"), read(folder/"RESULTS_FREEZE.json")
        verify(source["files"])
        verify(source.get("reference_files", {}))
        verify(result["files"])
        assert result["source_freeze_sha256"] == sha(folder/"FREEZE.json")
        audit = folder/"INDEPENDENT_AUDIT.json"
        assert sha(audit) == EXPECTED_AUDITS[family] and read(audit)["passed"]
        sources += [folder/"FREEZE.json", folder/"RESULTS_FREEZE.json", audit]
        sources += [ROOT/p for p in result["files"]]
        packages[family] = {"config": read(folder/"CONFIG.json")}
        if family != "NUMERIC":
            packages[family].update({key: read(folder/(name+".json")) for key, name in (
                ("contrasts", "CONTRASTS"), ("decision", "DECISION"), ("counts", "COUNTS"))})
    failed = BASE/"NUMERIC/AUDIT_FAILURE_SERIALIZATION"
    sources += [failed/"FAILURE.json", failed/"INDEPENDENT_AUDIT.partial.json"]
    assert read(failed/"FAILURE.json")["not_a_scientific_numerical_gate_failure"]
    refs = {d: read(BASE/"NUMERIC"/f"REFERENCE_d{d}.json") for d in (16, 64, 256)}
    assert all(r["precision"]["qualified"] and tuple(r["names"]) == NAMES for r in refs.values())
    assert len(packages["PROFILE"]["contrasts"]) == 16
    assert len(packages["HIGHK"]["contrasts"]) == 4
    summary = read(BASE/"NUMERIC/SUMMARY.json")
    assert summary["all_qualified"] and summary["primary_PMFS"] == 18 and summary["variant_PMFS"] == 72
    return {"packages": packages, "refs": refs, "summary": summary,
            "noise": read(BASE/"NUMERIC/NOISE_CONSTANTS.json"),
            "sources": {rel(p): sha(p) for p in sorted(set(sources))}}


def save_figure(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=200, facecolor="white")
    fig.savefig(OUT/(name+".svg"), facecolor="white", metadata={"Date": None})
    path = OUT/(name+".svg")
    write_text(path, path.read_text(encoding="utf-8"))
    plt.close(fig)


def risk_figure(data):
    dims = np.array([16, 64, 256])
    risks = np.array([data["refs"][int(d)]["risk_to_teacher"] for d in dims])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.7))
    fig.subplots_adjust(left=.09, right=.97, bottom=.23, top=.75, wspace=.30)
    fig.suptitle("Changing correlations can compensate for fixed low scales", y=.97, fontsize=14)
    ax = axes[0]
    handles = []
    for j, name in enumerate(NAMES):
        line, = ax.plot(dims, risks[:, j], marker="s" if name == "teacher" else "o",
                        label=name, color=COLORS[name], lw=1.7, ms=5)
        handles.append(line)
    ax.set_xscale("log", base=4)
    ax.set_yscale("symlog", linthresh=1e-10, linscale=.7)
    ax.set_ylim(-2.7e-11, .07)
    ax.set_xticks(dims, labels=dims)
    ax.set_yticks([0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2],
                  labels=["0", "$10^{-9}$", "$10^{-7}$", "$10^{-5}$", "$10^{-3}$", "$10^{-2}$"])
    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Total joint KL (nats; symlog includes zero)")
    ax.set_title("All six fixed models")
    ax.text(.05, .035, "Teacher = 0 at all three dimensions", transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=.2)
    ax = axes[1]
    comp = risks[:, 2]
    reference = comp[0]*(dims[0]/dims.astype(float))**2
    ax.loglog(dims, comp, "o-", color=COLORS["comp"], label="comp", lw=2)
    ax.loglog(dims, reference, "--", color="#6f6f6f", label=r"$d^{-2}$ guide, anchored at d=16")
    for d, value in zip(dims, comp):
        align = "left" if d == 16 else "right" if d == 256 else "center"
        offset = 6 if d == 16 else -6 if d == 256 else 0
        ax.annotate(f"{value:.4g}", (d, value), xytext=(offset, 9), textcoords="offset points",
                    ha=align, fontsize=9)
    ax.set_xticks(dims, labels=dims)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(12, 340)
    ax.set_ylim(comp[-1]*.45, comp[0]*3)
    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Compensated total joint KL (nats)")
    ax.set_title("Tiny risks resolved by independent numerical checks")
    ax.grid(alpha=.2)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .90), ncol=6, frameon=False)
    fig.text(.5, .065, "Oracle loadings use teacher information. No model was fitted or selected.\n"
             "The rate guide is descriptive; these three values do not prove a sharp KL rate.",
             ha="center", fontsize=9, color="#444444")
    save_figure(fig, FIGURES[0])


def score_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), sharey=True)
    fig.subplots_adjust(left=.09, right=.97, bottom=.27, top=.73, wspace=.15)
    fig.suptitle("A fixed pair: raw-score preference changes with particle budget", y=.97, fontsize=14)
    for ax, d in zip(axes, (16, 64)):
        ax.axhline(0, color="#555555", lw=1)
        ref = data["refs"][d]["risk_to_teacher"]
        riskdiff = ref[4]-ref[2]
        ax.axhline(riskdiff, color="#777777", linestyle="--", lw=1.3)
        ax.text(.96, .96, f"Model KL difference = +{riskdiff:.5f}", transform=ax.transAxes,
                ha="right", va="top", color="#555555", fontsize=9)
        for family, color, marker in (("PROFILE", "#337ca5", "o"), ("HIGHK", "#d07a2d", "s")):
            rows = [r for r in data["packages"][family]["contrasts"]
                    if r["d"] == d and r["comparison"] == "over_minus_comp"]
            rows.sort(key=lambda r: r["K"])
            xx = [r["K"] for r in rows]
            yy = [r["mean"] for r in rows]
            err = [[r["mean"]-r["lower"] for r in rows], [r["upper"]-r["mean"] for r in rows]]
            ax.errorbar(xx, yy, yerr=err, color=color, fmt=marker, capsize=4, ms=6, lw=1.6)
            ax.plot(xx, yy, color=color, alpha=.45, lw=1)
            for r in rows:
                if not r["precision_pass"]:
                    ax.plot(r["K"], r["mean"], "x", color="red", ms=11)
        ax.plot([1], [0], "D", color="#222222", ms=5)
        ax.set_xscale("log")
        ks = sorted({1, 10, d, 256, 4096})
        ax.set_xticks(ks, labels=[str(k) for k in ks])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlim(.65, 6500)
        ax.set_ylim(-.074, .043)
        ax.set_xlabel("Particles per bank K (log scale)")
        ax.set_title(f"d = {d}")
        ax.grid(axis="y", alpha=.2)
        if d == 16:
            ax.annotate("K=4096: reversal gate passed", (4096, .0276249488),
                        xytext=(.20, .70), textcoords="axes fraction", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#666666"})
        else:
            ax.annotate("K=4096: CI crosses zero\nPrimary reversal gate failed", (4096, .0021276133),
                        xytext=(.11, .65), textcoords="axes fraction", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#666666"})
    axes[0].set_ylabel(r"$F_K(\mathrm{over})-F_K(\mathrm{comp})$ (total nats)")
    legend = [Line2D([0], [0], marker="D", color="none", markerfacecolor="#222222",
                     label="K=1: analytic tie"),
              Line2D([0], [0], marker="o", color="#337ca5", label="PROFILE: 16-contrast t family"),
              Line2D([0], [0], marker="s", color="#d07a2d", label="HIGHK: separate 4-contrast t family")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.51, .90), ncol=3,
               frameon=False, fontsize=9)
    fig.text(.5, .105, "Negative score difference favors the worse-risk over model; positive favors comp.\n"
             "Each package uses its own simultaneous 95% approximate t intervals; no pooled 95% claim.",
             ha="center", fontsize=9)
    fig.text(.5, .03, "PROFILE: 8 streams x 2048 banks per dimension. HIGHK: 8 x 64, independently frozen follow-up.\n"
             "K=1 is an exact analytic identity, not a zero-variance Monte Carlo inference. No global optimum is claimed.",
             ha="center", fontsize=8.7, color="#444444")
    save_figure(fig, FIGURES[1])


def noise_rows(data):
    constants = data["noise"]["primary"]
    teacher = constants["negative_teacher_limit_variance"]
    tau2 = constants["emission_noise_variance_tau2"]
    rows = []
    for h in np.linspace(0, 2.5, 501):
        p = math.erf(float(h)/math.sqrt(2*teacher))
        q = math.erf(float(h)/math.sqrt(2*tau2))
        rows.append({"h": float(h), "teacher_limit_probability": p,
                     "asymptotic_nearopt_probability_ceiling": q,
                     "asymptotic_excess_Brier_lower": (p-q)**2})
    return rows


def noise_figure(data, rows):
    constants = data["noise"]["primary"]
    h = np.array([r["h"] for r in rows])
    p = np.array([r["teacher_limit_probability"] for r in rows])
    q = np.array([r["asymptotic_nearopt_probability_ceiling"] for r in rows])
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    fig.subplots_adjust(left=.11, right=.96, bottom=.30, top=.76)
    fig.suptitle("Negative teacher: an asymptotic interval-probability obstruction", y=.97, fontsize=14)
    ax.plot(h, p, color="#347c9b", lw=2.2, label=r"Teacher limit: $2\Phi(h/\sqrt{V_*^-})-1$")
    ax.plot(h, q, color="#c17437", lw=2.2, linestyle="--",
            label=r"Near-optimal model ceiling: $2\Phi(h/\tau_a)-1$")
    ax.fill_between(h, q, p, color="#5a9597", alpha=.14)
    ax.axvline(1, color="#777777", lw=.9, linestyle=":")
    i = next(i for i, row in enumerate(rows) if row["h"] == 1.)
    ax.scatter([1, 1], [p[i], q[i]], color=["#347c9b", "#c17437"], zorder=4)
    ax.annotate(f"h=1: teacher {p[i]:.5f}\nceiling {q[i]:.5f}\ngap {p[i]-q[i]:.5f}",
                xy=(1, (p[i]+q[i])/2), xytext=(1.37, .39), fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "#777777"})
    ax.text(.025, .93, rf"$V_*^- = {constants['negative_teacher_limit_variance']:.6f}$"+"\n"
            +rf"$\tau_a^2 = {constants['emission_noise_variance_tau2']:.6f}$", transform=ax.transAxes,
            va="top", fontsize=10)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Interval half-width h for $S_d=d^{-1/2}\sum_iY_i$")
    ax.set_ylabel(r"Probability of $|S_d|\leq h$")
    ax.grid(alpha=.18)
    fig.legend(loc="upper center", bbox_to_anchor=(.52, .90), ncol=1, frameon=False, fontsize=10)
    fig.text(.5, .13, "CHANNEL H11: limsup over population near-optimal models, with log K=o(d), epsilon=o(d).\n"
             "Unbounded means are allowed; zero teacher marginals and fixed spectral / scale bounds remain required.",
             ha="center", fontsize=9)
    fig.text(.5, .045, "This is a normal-limit display, not a finite-d certified bound.\n"
             "Curves evaluate erf from existing numerically qualified constants; no new quadrature or fitting.",
             ha="center", fontsize=9, color="#444444")
    save_figure(fig, FIGURES[2])


def output_rows(data):
    numeric, events, spectra, precision, contrasts, primary = [], [], [], [], [], []
    for d, ref in data["refs"].items():
        for j, name in enumerate(NAMES):
            specs = ref["spectra"][j]
            numeric.append({"d": d, "name": name, "scale_A": ref["params"][j]["scales"][0],
                            "scale_B": ref["params"][j]["scales"][1],
                            "lambda_A": ref["params"][j]["lambda"][0],
                            "lambda_B": ref["params"][j]["lambda"][1],
                            "joint_KL": ref["risk_to_teacher"][j],
                            "min_eigenvalue": specs["minimum"], "max_eigenvalue": specs["maximum"],
                            "trace": specs["trace"]})
            for k, event in enumerate(ref["event_names"]):
                events.append({"d": d, "name": name, "event": event,
                               "probability": ref["event_probability"][j][k],
                               "excess_Brier": ref["event_brier"][j][k],
                               "binary_KL": ref["event_logloss"][j][k]})
            spectra += [{"d": d, "name": name, "index": k, "eigenvalue": value}
                        for k, value in enumerate(specs["eigenvalues"])]
        p = ref["precision"]
        precision.append({"d": d, **{k: p[k] for k in (
            "mass_error", "PMF_L1_variant_difference", "log_PMF_variant_difference",
            "complement_symmetry_error", "comp_KL_relative_span", "event_probability_span",
            "event_risk_span", "quad_max_log_difference", "qualified")},
            "maximum_KL_span": max(p["KL_cross_variant_spans"]),
            "comp_KL_span": p["KL_cross_variant_spans"][2],
            "maximum_Decimal80_absolute_difference": max(r["absolute_difference"] for r in p["Decimal80_audits"])})
    for family in ("PROFILE", "HIGHK"):
        package = data["packages"][family]
        for row in package["contrasts"]:
            contrasts.append({"family": family, "family_size": package["config"]["family"],
                              **{k: row[k] for k in ("d", "K", "comparison", "mean", "SE", "lower", "upper",
                                                     "reference_risk_difference", "precision_pass")}})
        for row in package["decision"]["primary"]:
            key = "score_upper" if family == "PROFILE" else "score_lower"
            primary.append({"family": family, "d": row["d"], "K": row["K"],
                            "score_bound_name": key, "score_bound": row[key],
                            "reference_risk_difference": row["risk_difference"], "passed": row["passed"]})
    return {"NUMERIC_MODELS.csv": numeric, "EVENTS.csv": events, "SPECTRA.csv": spectra,
            "NUMERIC_PRECISION.csv": precision, "SCORE_CONTRASTS.csv": contrasts, "PRIMARY_GATES.csv": primary}


def report_numeric(data, tables):
    models = tables["NUMERIC_MODELS.csv"]
    rows = []
    for d in (16, 64, 256):
        ref = data["refs"][d]
        rows.append([d]+[n(x) for x in ref["risk_to_teacher"]])
    text = """# NUMERIC：固定相关结构与补偿候选的完整结果

六个固定模型在三个维数都通过原定数值资格门。补偿候选的联合 KL 从 d16 的 7.37697e−8 降到 d256 的 2.74462e−10；固定 R、尺度降到 (1,1) 的 KL 仍约 .013。补偿使用教师已知参数，是表示能力的 oracle 构造，未做学习或极小点搜索。三个维数呈现与 d⁻² 相符的描述性趋势；这些点不能独立证明最优收敛率。

教师是正相关、零均值、两已知等大组、尺度 (6,2)，rho=.9/(d−1)。五候选尺度均为 (1,1)。fixedRfloor 保持教师 R；comp 的两组载荷为 sqrt(rho) alpha(s_g)/alpha(1)；half/over 为其 .5/1.5 倍；independent 为零载荷。所有模型仍是一个 Gaussian 潜变量加条件独立 Bernoulli 发射。

## 全部 18 个主 PMF 的联合风险

单位是整向量总 nats，未除以 d。teacher=0 是模型自比恒等式，图中保留零点。

"""+table(["d", *NAMES], rows)+"\n\n"
    text += "## 全部协方差和参数摘要\n\n"+table(
        ["d", "模型", "尺度 A/B", "载荷 A/B", "最小特征值", "最大特征值", "trace"],
        [[r["d"], r["name"], f"{r['scale_A']:g}/{r['scale_B']:g}",
          f"{n(r['lambda_A'])}/{n(r['lambda_B'])}", n(r["min_eigenvalue"]),
          n(r["max_eigenvalue"]), n(r["trace"])] for r in models])+"\n\n"
    text += "全模型满足 0.1 I ≤ Sigma ≤ 121.6 I。完整各特征值见 [SPECTRA.csv](../../../artifacts/covariance_boundary/figures/SPECTRA.csv)，解析两组分解与直接 eigvalsh 交叉核验。正族补偿不是通过接近奇异协方差实现。\n\n"
    text += "## 原预定数值门与实测误差\n\n"
    metrics = [("原质量误差", "mass_error", 1e-12), ("PMF L1 变体差", "PMF_L1_variant_difference", 1e-8),
               ("全格 log-PMF 差", "log_PMF_variant_difference", 1e-6),
               ("补对称误差", "complement_symmetry_error", 1e-12),
               ("最大 KL 交叉变体极差", "maximum_KL_span", 1e-10),
               ("comp KL 极差", "comp_KL_span", 1e-12),
               ("comp 相对极差", "comp_KL_relative_span", 1e-3),
               ("事件概率极差", "event_probability_span", 1e-10),
               ("事件风险极差", "event_risk_span", 1e-10),
               ("独立私噪声 quad log 差", "quad_max_log_difference", 1e-9),
               ("Decimal80 绝对差", "maximum_Decimal80_absolute_difference", 1e-14)]
    text += table(["检查", "门", "d16", "d64", "d256"],
                  [[label, n(gate)]+[n(r[key]) for r in tables["NUMERIC_PRECISION.csv"]]
                   for label, key, gate in metrics])+"\n\n"
    text += """所有原门均通过。另保留 comp Decimal 相对差 ≤1e−6、谱公式差 ≤1e−10、原 d16/64 参考锚差 ≤1e−10；全部细项在冻结 REFERENCE JSON。主计算使用归一 PMF 的 generalized KL，近等概率采用稳定级数，另用 Decimal80 对同一浮点 PMF 重新求值。Decimal 核验不提升求积精度；四变体和独立自适应 quad 也不是严格舍入/积分区间证书。d256 的最大交叉变体误差主要出现在粗变体，原所有变体均保留。

完整事件概率、额外 Brier 和二元 KL 共 54 行见 [EVENTS.csv](../../../artifacts/covariance_boundary/figures/EVENTS.csv)。事件为 S=2(N_A+N_B)−d 的 |S|≤sqrt(d)、S>sqrt(d)、S<−sqrt(d)。它们不替代完整联合 KL。

## 噪声常数、资源与审计

"""
    constants = data["noise"]["primary"]
    text += f"tau_a²=2 alpha(1)={n(constants['emission_noise_variance_tau2'])}，负教师总自旋极限方差 V*⁻={n(constants['negative_teacher_limit_variance'])}，差={n(constants['tau2_minus_teacher_variance'])}。这里的负教师与上表正教师不同。alpha 沿用旧已合格 GH2048 主值，GH1024/quad 敏感性原样保留，0 新 alpha 积分。tau_a² 不能再平方。\n\n"
    elapsed = sum(r["elapsed_seconds"] for r in data["summary"]["rows"])
    text += f"原生产工作为 18 主 PMF、72 积分变体、270 固定 z 检查（正负概率合计 540 次 scalar quad）、15 次 Decimal80 候选 KL 核验，三维数记录耗时合计 {elapsed:.3f} 秒；0 新观察、随机粒子、拟合。独立审计的额外确定性积分和核验另计，不重复算作新科学模型。\n\n"
    text += "NUMERIC 最终独立审计通过。首次审计因 NumPy boolean JSON 序列化中断，保留 [FAILURE.json](../../../artifacts/covariance_boundary/NUMERIC/AUDIT_FAILURE_SERIALIZATION/FAILURE.json) 与不完整收据；这是审计写出故障，不是生产数值门失败，原冻结和数值规则未改。\n\n"
    text += "完整来源与最终审计哈希见 [RESULTS](RESULTS.md)；图文封存在 `artifacts/covariance_boundary/figures/MANIFEST.json`。\n"
    write_text(DOC/"NUMERIC_RESULTS.md", text)


def report_results(data, tables, curves):
    primary = tables["PRIMARY_GATES.csv"]
    text = """# 固定模型风险、有限粒子排序与发射噪声边界

正族中，改变相关结构能把固定低尺度模型的观察联合 KL 降至近零；但原 raw 有限粒子评分在所测小 K 下偏好过度补偿的更坏模型。提高 K 后，d16 预定 pair 的排序反转；d64、K4096 的区间仍跨零，原主门失败。另一个解析方向是 CHANNEL 的发射噪声定理：在零教师边际、固定正尺度下限和维数统一谱界下，允许无界均值和全部相关结构可训练，仍可对满足信号条件的负族约束所有总体近优点的渐近事件预测。

这些是六个固定候选与一个预定 pair 的评价，不是已学得模型、连续全局极小点或 SGD 结论。下表和图不将正族反例扩成负族定理的反例。

## 1 完整联合风险与补偿结构

![全部模型与微小补偿风险](../../../artifacts/covariance_boundary/figures/compensation_risk.png)

全部 18 模型的风险、参数、谱与原数值误差门见 [NUMERIC_RESULTS](NUMERIC_RESULTS.md)。图中 teacher=0 明确保留；d⁻² 只是锚定 d16 的参考斜率，不是从三个点认证的普遍收敛率。所有补偿载荷用已知教师信息确定。

## 2 原始总体评分的固定 pair 比较

![固定 pair 随 K 的排序](../../../artifacts/covariance_boundary/figures/score_pair_across_K.png)

F_K 是 prior 等权 K 粒子在 log 内、独立 bank 平均在 log 外的原始总体目标。按完整教师 count PMF 加权 canonical patterns，依赖两组内交换性在期望上恢复该 raw 目标；没有把每个 raw 粒子改成 group-count Rao–Blackwell 分量。控制变量降低数值方差，不改总体目标。K1 的同 floor 解析恒等式来自相同单坐标潜分布，与 Monte Carlo 区间无关。

PROFILE 原 16 项家族为 8 流×2048 bank/维数；HIGHK 看到 PROFILE 后另立计划和独立随机流，为 8 流×64 bank/维数、4 项家族。两包各自 Bonferroni 同时 95% t 近似区间，不合并成跨包总体 95% 保证，不把 B×K 并成一个 log 目标。

### 全部原主门

PROFILE：要求 over−comp 的评分上端 <−.001、参考风险差 >.001 且原精度通过。HIGHK：仅 K4096 是主门，要求评分下端 >.001、风险差 >.001 且原精度通过。两个方向不同，HIGHK 不改写 PROFILE 的原判定。

"""+table(["家族", "d", "K", "评分区间端点", "风险差", "原主门"],
                  [[r["family"], r["d"], r["K"], f"{r['score_bound_name']}={n(r['score_bound'])}",
                    n(r["reference_risk_difference"]), "通过" if r["passed"] else "失败"] for r in primary])+"\n\n"
    text += "d64 K4096 点估计虽为正，区间跨 0，既未确认正向排序，也未证明仍反向；停止在原预算，不追加 K 或 bank。所有 20 项精度门通过（SE≤.003）；没有被隐藏的精度失败。主门合计 5/6 仅是两包判定的计数，不构成一个新统计家族。\n\n"
    for family in ("PROFILE", "HIGHK"):
        text += f"### {family} 完整置信区间表\n\n"
        rows = [r for r in tables["SCORE_CONTRASTS.csv"] if r["family"] == family]
        text += table(["d", "K", "差（前−后）", "mean", "SE", "lower", "upper", "参考风险差", "精度"],
                      [[r["d"], r["K"], r["comparison"], *[n(r[k]) for k in ("mean", "SE", "lower", "upper", "reference_risk_difference")],
                        "通过" if r["precision_pass"] else "失败"] for r in rows])+"\n\n"
    text += "## 3 负教师的渐近区间概率边界\n\n![渐近噪声概率上界](../../../artifacts/covariance_boundary/figures/negative_noise_limit.png)\n\n"
    at1 = next(r for r in curves if r["h"] == 1.)
    text += f"图按冻结常数直接计算正态 CDF（erf），没有新增求积。h=1 时教师极限概率为 {n(at1['teacher_limit_probability'])}，总体近优模型的渐近上限为 {n(at1['asymptotic_nearopt_probability_ceiling'])}，相应渐近额外 Brier 下界为 {n(at1['asymptotic_excess_Brier_lower'])}。常数本身仅经数值资格核验，图不是严格有限维机器证书。\n\n"
    text += "最终理论引用 [CHANNEL_THEORY](CHANNEL_THEORY.md) H1–H12 与 [CHANNEL_REVIEW](CHANNEL_REVIEW.md)。其中 J=g+2l 的凸性给常数 2 的通道恒等式，候选均值可在全实轴；谱上界/正下界、尺度下限、零教师边际、独立 Bernoulli 发射和 logK=o(d)、近优容差 epsilon=o(d) 仍不可省略。曲线画 H11 的 limsup 上限与负教师 CLT，不用它替换 H10 的有限维误差项。原 [NOISE_THEORY](NOISE_THEORY.md) 的有界均值证明作为历史保留，不能替代最终更强版本。\n\n"
    text += "## 4 资源、来源与审计\n\n"
    ledger = []
    for family in ("PROFILE", "HIGHK"):
        c = data["packages"][family]["counts"]
        ledger.append([family, c["banks"], c["unique_base_prior_vectors"], c["base_gaussian_scalars"],
                       c["count_particle_candidate_terms"], c["logsumexp_terms"], c["new_teacher_labels"], c["new_fits"]])
    text += table(["包", "banks", "不同基准潜向量", "基准 Gaussian 标量", "候选 count×particle 项", "logsumexp 项", "新标签", "拟合"], ledger)+"\n\n"
    text += "本图表整理新增科学积分/随机样本/拟合均为 0；只读冻结 JSON、确定性表格汇总和 erf 显示。NUMERIC 的生产与审计资源分账见其专页。\n\n"
    text += table(["来源", "SHA256"], [[f"{family}/INDEPENDENT_AUDIT.json", f"`{digest}`"]
                                         for family, digest in EXPECTED_AUDITS.items()]
                  +[["CHANNEL_THEORY.md（最终无界均值版本）", f"`{CHANNEL_SHA}`"]])+"\n\n"
    text += "三个最终独立审计均通过；NUMERIC 首次序列化故障和不完整收据作为来源输入绑定，并未抹去。`SOURCE_HASHES.json` 绑定所有绘图来源、最终审计和生成器；`MANIFEST.json` 在三图实际视觉核验后绑定图、完整 CSV、本两份结果文档。默认 `check` 只读，不重放采样、拟合或积分。\n"
    write_text(DOC/"RESULTS.md", text)


def build():
    assert not (OUT/"MANIFEST.json").exists(), "sealed figures cannot be overwritten"
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "covariance-boundary"})
    tables = output_rows(data)
    curves = noise_rows(data)
    for name, rows in tables.items():
        csv_file(name, rows)
    csv_file("NOISE_LIMIT_CURVES.csv", curves)
    risk_figure(data)
    score_figure(data)
    noise_figure(data, curves)
    report_numeric(data, tables)
    report_results(data, tables, curves)
    write_json(OUT/"SOURCE_HASHES.json", data["sources"])
    files = [p for p in OUT.iterdir() if p.is_file() and p.name not in
             ("BUILD_RECEIPT.json", "VISUAL_QA.json", "MANIFEST.json")]
    files += [DOC/"RESULTS.md", DOC/"NUMERIC_RESULTS.md"]
    write_json(OUT/"BUILD_RECEIPT.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(Path(__file__)),
        "files": {rel(p): sha(p) for p in files},
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "figures": list(FIGURES), "primary_PMFS": 18, "variant_PMFS": 72,
        "original_quad_probability_calls": 540, "all_contrasts": 20, "all_primary_gates": 6,
        "precision_failures": sum(not r["precision_pass"] for r in tables["SCORE_CONTRASTS.csv"]),
        "primary_failures": sum(not r["passed"] for r in tables["PRIMARY_GATES.csv"]),
        "new_scientific_integrals": 0, "new_RNG_samples": 0, "new_fits": 0,
        "normal_CDF_display_only": True, "visual_qa_complete": False})
    print("Built three figures and complete readable tables; not sealed until visual QA.")


def seal(visual_qa_complete):
    assert visual_qa_complete, "actual visual QA must precede seal"
    assert not (OUT/"MANIFEST.json").exists(), "refuse reseal"
    receipt = read(OUT/"BUILD_RECEIPT.json")
    assert receipt["source_sha256"] == sha(Path(__file__))
    verify(receipt["files"])
    verify(read(OUT/"SOURCE_HASHES.json"))
    write_json(OUT/"VISUAL_QA.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "complete": True,
        "viewed_PNGs": {name+".png": sha(OUT/(name+".png")) for name in FIGURES},
        "scope": "All three rendered PNGs viewed; labels, CIs, zero risk, family separation and limit caveat checked."})
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != "MANIFEST.json"]
    paths += [DOC/"RESULTS.md", DOC/"NUMERIC_RESULTS.md", Path(__file__).resolve()]
    write_json(OUT/"MANIFEST.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(Path(__file__)),
        "files": {rel(p): sha(p) for p in paths}, "sources": read(OUT/"SOURCE_HASHES.json"),
        "visual_qa_complete": True, "new_scientific_computation": False})
    print(json.dumps({"sealed": True, "manifest_sha256": sha(OUT/"MANIFEST.json")}))


def check():
    manifest = read(OUT/"MANIFEST.json")
    verify(manifest["files"])
    verify(manifest["sources"])
    assert manifest["visual_qa_complete"] and not manifest["new_scientific_computation"]
    receipt = read(OUT/"BUILD_RECEIPT.json")
    assert receipt["all_contrasts"] == 20 and receipt["primary_failures"] == 1
    assert receipt["precision_failures"] == 0 and receipt["primary_PMFS"] == 18
    print(json.dumps({"passed": True, "read_only": True, "new_scientific_computation": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("build", "seal", "check"))
    parser.add_argument("--visual-qa-complete", action="store_true")
    args = parser.parse_args()
    if args.phase == "build":
        build()
    elif args.phase == "seal":
        seal(args.visual_qa_complete)
    else:
        check()
