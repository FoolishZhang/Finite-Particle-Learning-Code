"""Figures/readable tables from completed finite_learning results only.

Build never fits, draws, integrates, or solves the frozen mean-display root.
Do not build before ROOT authorizes reading the completed RUN outputs.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/finite_learning"
RUN = BASE/"RUN"
MEAN = BASE/"MEAN_DISPLAY"
REF = BASE/"REF"
OUT = BASE/"figures"
DOC = ROOT/"docs/08_research_研究工作/有限样本学习与均值自由度_20261001"
FIGURE_NAMES = ("risk_by_N_B", "risk_contrasts", "event_probabilities", "mean_compensation")
METHODS = ("raw", "col_d", "col_4d", "MLE")
LABELS = {"raw": "raw (K=d)", "col_d": "col (K=d)",
          "col_4d": "col (K=4d; auxiliary)", "MLE": "Numerical MLE"}
COLORS = {"raw": "#c66a35", "col_d": "#2672a2", "col_4d": "#705898", "MLE": "#37846b"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_text(path, value):
    value = "\n".join(line.rstrip() for line in value.splitlines()).rstrip()+"\n"
    Path(path).write_text(value, encoding="utf-8", newline="\n")


def verify(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


def tag(row):
    if row["method"] == "col":
        return "col_d" if row["K"] == row["d"] else "col_4d"
    return row["method"]


def select_models(models, d, N, B, method):
    selected = [r for r in models if r["d"] == d and r["N"] == N and tag(r) == method
                and (method == "MLE" or r["B"] == B)]
    selected.sort(key=lambda r: r["stream"])
    assert len(selected) == 8 and [r["stream"] for r in selected] == list(range(8))
    return selected


def source_paths():
    paths = [Path(__file__).resolve()]
    for folder in (REF, RUN, MEAN):
        paths.extend((folder/"FREEZE.json", folder/"RESULTS_FREEZE.json"))
    paths += [RUN/name for name in (
        "CONFIG.json", "MODEL_RESULTS.json", "RISK_TABLES.json", "RISK_CONTRASTS.json",
        "SCORE_CONTRASTS.json", "FLOOR_PROBABILITY_BOUNDS.json", "DECISION.json", "COUNTS.json")]
    paths += [REF/"QUALIFICATION.json", MEAN/"RESULTS.json"]
    return paths


def load():
    assert read(RUN/"STATE.json")["status"] == "complete", "RUN must be complete before build"
    for folder in (REF, RUN, MEAN):
        source = read(folder/"FREEZE.json")
        result = read(folder/"RESULTS_FREEZE.json")
        verify(source["files"])
        verify(result["files"])
        assert result["source_freeze_sha256"] == sha(folder/"FREEZE.json")
    assert read(REF/"QUALIFICATION.json")["all_passed"]
    models = read(RUN/"MODEL_RESULTS.json")
    risk = read(RUN/"RISK_CONTRASTS.json")
    scores = read(RUN/"SCORE_CONTRASTS.json")
    assert len(models) == 224 and len(risk) == 32 and len(scores) == 24
    return {
        "config": read(RUN/"CONFIG.json"), "models": models, "risk": risk, "scores": scores,
        "risk_tables": read(RUN/"RISK_TABLES.json"), "floor": read(RUN/"FLOOR_PROBABILITY_BOUNDS.json"),
        "decision": read(RUN/"DECISION.json"), "counts": read(RUN/"COUNTS.json"),
        "mean": read(MEAN/"RESULTS.json"),
    }


def save_figure(fig, name):
    for suffix in ("png", "svg"):
        path = OUT/f"{name}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white")
        if suffix == "svg":
            write_text(path, path.read_text("utf-8"))
    plt.close(fig)


def risk_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.9))
    fig.subplots_adjust(left=.085, right=.985, bottom=.11, top=.82, hspace=.34, wspace=.28)
    fig.suptitle("Finite-label learning: all N and bank budgets", y=.985, fontsize=14)
    upper_limit = max(float(r["reference_joint_KL"]) for r in data["models"])*1.65
    for di, d in enumerate((16, 64)):
        for ni, N in enumerate((256, 4096)):
            ax = axes[di, ni]
            for mi, method in enumerate(METHODS):
                means = []
                xs = np.array([0., 1.])+(mi-1.5)*.035
                for bi, B in enumerate((16, 128)):
                    rows = select_models(data["models"], d, N, B, method)
                    values = np.asarray([r["reference_joint_KL"] for r in rows])
                    means.append(float(values.mean()))
                    jitter = (np.arange(8)-3.5)*.006
                    ax.scatter(xs[bi]+jitter, values, s=13, alpha=.45,
                               color=COLORS[method], zorder=2)
                ax.plot(xs, means, color=COLORS[method], marker="o", markersize=5,
                        lw=1.7, ls="--" if method == "col_4d" else "-", label=LABELS[method])
            ax.set_title(f"d={d}, N={N}")
            ax.set_xticks([0, 1], ["B=16", "B=128"])
            ax.set_xlim(-.16, 1.16)
            ax.set_yscale("symlog", linthresh=1e-4, linscale=.7)
            ax.set_ylim(-1e-5, upper_limit)
            ax.axhline(0, color="#777777", lw=.6)
            ax.grid(alpha=.17, axis="y", which="both")
            if ni == 0:
                ax.set_ylabel("Total joint KL (nats)\nlinear near zero; log outside")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .947),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .03, "Dots: 8 independent training streams. Lines: means. MLE is reused across B.\n"
             "Same-K comparison is raw versus col(K=d); col(K=4d) uses additional particles.",
             ha="center", fontsize=9)
    save_figure(fig, "risk_by_N_B")


def contrast_figure(data):
    names = ("raw_minus_col_d", "col_d_minus_MLE", "col_4d_minus_MLE", "col_d_minus_col_4d")
    titles = ("raw - col(K=d)", "col(K=d) - MLE", "col(K=4d) - MLE", "col(K=d) - col(K=4d)")
    cells = [(d, N, B) for d in (16, 64) for N in (256, 4096) for B in (16, 128)]
    fig, axes = plt.subplots(1, 4, figsize=(15., 6.5), sharey=True)
    fig.subplots_adjust(left=.17, right=.985, bottom=.18, top=.80, wspace=.32)
    fig.suptitle("All 32 frozen risk contrasts, including precision failures", y=.98, fontsize=14)
    for column, (name, title) in enumerate(zip(names, titles)):
        ax = axes[column]
        for row, cell in enumerate(cells):
            item = next(r for r in data["risk"] if (r["d"], r["N"], r["B"]) == cell
                        and r["comparison"] == name)
            color = "#2672a2" if item["precision_pass"] else "#b33939"
            marker = "o" if item["precision_pass"] else "x"
            ax.errorbar(item["mean"], row,
                        xerr=np.array([[item["mean"]-item["lower"]], [item["upper"]-item["mean"]]]),
                        fmt=marker, color=color, capsize=3, lw=1.15, markersize=5)
        ax.axvline(0, color="#555555", lw=.8)
        if column == 0:
            ax.axvline(data["config"]["material_improvement"], ls=":", color="#ad7b28", lw=1)
        elif column in (1, 2):
            ax.axvline(data["config"]["noninferiority_margin"], ls=":", color="#ad7b28", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Paired total KL difference (nats)", fontsize=9)
        ax.grid(alpha=.15, axis="x")
    axes[0].set_yticks(range(8), [f"d={d}; N={N}; B={B}" for d, N, B in cells], fontsize=9)
    axes[0].invert_yaxis()
    handles = [
        Line2D([], [], marker="o", color="#2672a2", label="Precision qualified", lw=1),
        Line2D([], [], marker="x", color="#b33939", label="Precision failed (retained)", lw=1),
        Line2D([], [], color="#ad7b28", ls=":", label="Prespecified material / noninferiority threshold")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3,
               frameon=False, fontsize=9)
    fig.text(.5, .05, "Original simultaneous t approximation: risk family 32, alpha=.025; 8 streams.\n"
             "Zero-width intervals can arise from identical discrete selections and are not strict population equivalence certificates.",
             ha="center", fontsize=9)
    save_figure(fig, "risk_contrasts")


def event_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.18, top=.77, wspace=.20)
    fig.suptitle("High-N/high-B event probabilities (reference, not new test draws)", y=.985, fontsize=13)
    for di, d in enumerate((16, 64)):
        ax = axes[di]
        teacher = np.asarray(data["risk_tables"][str(d)]["teacher_events"])
        ax.plot(range(3), teacher, marker="D", color="#222222", ls="--", lw=1.5, label="Teacher")
        for mi, method in enumerate(METHODS):
            rows = select_models(data["models"], d, 4096, 128, method)
            values = np.asarray([r["event_probabilities"] for r in rows])
            x = np.arange(3)+(mi-1.5)*.035
            for s in range(8):
                ax.scatter(x+(s-3.5)*.004, values[s], color=COLORS[method], alpha=.4, s=12)
            ax.plot(x, values.mean(axis=0), marker="o", color=COLORS[method], lw=1.3,
                    ls="--" if method == "col_4d" else "-", label=LABELS[method])
        ax.set_title(f"d={d}; N=4096; B=128")
        ax.set_xticks(range(3), ["Central", "Upper tail", "Lower tail"])
        ax.set_xlim(-.20, 2.2)
        ax.grid(alpha=.15, axis="y")
        ax.set_ylabel("Event probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .055, "Central: |sum Y| <= sqrt(d); upper/lower tails partition the complement.\n"
             "All fitted RUN models have zero means. Dots are selected-model probabilities, not independent event samples.",
             ha="center", fontsize=9)
    save_figure(fig, "event_probabilities")


def mean_figure(data):
    r = data["mean"]
    x = np.asarray(r["beta"])
    center, left, right = [np.asarray(r[k]) for k in ("center", "left", "right")]
    assert len(x) == len(center) == len(left) == len(right) == r["analytic_grid_points"] == 1601
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(left=.08, right=.985, bottom=.18, top=.76, wspace=.24)
    fig.suptitle("Normal-limit illustration - not a finite-d fitted model", y=.985, fontsize=14)
    beta = r["matched_beta"]
    axes[0].plot(x, center, color="#2672a2", label="Candidate central probability")
    axes[0].axhline(r["teacher_center"], color="#333333", ls="--")
    axes[0].scatter([beta], [r["candidate_center"]], color="#2672a2", s=35, zorder=4)
    axes[0].annotate(f"Central probability matched\nmean drift = {beta:.3f}",
                     (beta, r["candidate_center"]), xytext=(.28, .10), textcoords="axes fraction",
                     arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=9)
    axes[0].set(title="A mean shift can match the central event",
                xlabel="Candidate mean drift of S", ylabel="Probability of |S| <= 1")
    axes[1].plot(x, left, color="#705898", label="Candidate left tail")
    axes[1].plot(x, right, color="#37846b", label="Candidate right tail")
    axes[1].axhline(r["teacher_each_tail"], color="#333333", ls="--")
    axes[1].axvline(beta, color="#888888", ls=":", lw=1)
    axes[1].scatter([beta, beta], [r["candidate_left"], r["candidate_right"]],
                    color=["#705898", "#37846b"], s=35, zorder=4)
    axes[1].set(title="The same shift leaves asymmetric tail errors",
                xlabel="Candidate mean drift of S", ylabel="Tail probability")
    for ax in axes:
        ax.set_xlim(-1.2, 1.2)
        ax.grid(alpha=.17)
    handles = [
        Line2D([], [], color="#2672a2", label="Candidate central"),
        Line2D([], [], color="#705898", label="Candidate left tail"),
        Line2D([], [], color="#37846b", label="Candidate right tail"),
        Line2D([], [], color="#333333", ls="--", label="Teacher references")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91),
               ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .05, "Redrawn from frozen MEAN_DISPLAY arrays; no scalar root, integration, or fitted model regenerated.",
             ha="center", fontsize=9)
    save_figure(fig, "mean_compensation")


def number(value):
    return f"{float(value):.8g}"


def table(headers, rows):
    lines = ["|"+"|".join(headers)+"|", "|"+"|".join("---" for _ in headers)+"|"]
    for row in rows:
        lines.append("|"+"|".join(str(v).replace("|", r"\|") for v in row)+"|")
    return "\n".join(lines)


def contrast_table(rows):
    return table(["d", "N", "B", "对比", "mean", "SE", "区间", "SE门", "精度"],
                 [[r["d"], r["N"], r["B"], r["comparison"], number(r["mean"]), number(r["SE"]),
                   f'[{number(r["lower"])},{number(r["upper"])}]', number(r["SE_gate"]),
                   "通过" if r["precision_pass"] else "**失败**"] for r in rows])


def readable(data):
    summaries = []
    for d in (16, 64):
        for N in (256, 4096):
            for B in (16, 128):
                for method in METHODS:
                    rows = select_models(data["models"], d, N, B, method)
                    summaries.append({
                        "d": d, "N": N, "B": B, "method": method, "streams": 8,
                        "mean_joint_KL": float(np.mean([r["reference_joint_KL"] for r in rows])),
                        "mean_event_brier": np.mean([r["event_brier"] for r in rows], axis=0).tolist(),
                        "selected_scales": [r["scales"] for r in rows],
                        "floor_count": sum(r["scales"] == [1., 1.] for r in rows)})
    write_json(OUT/"SUMMARY.json", summaries)
    full_rows = [[r["d"], r["stream"], r["method"], r["N"], r["B"], r["K"],
                  r["scales"], number(r["reference_joint_KL"])]
                 for r in sorted(data["models"], key=lambda r: (
                     r["d"], r["stream"], r["N"], str(r["method"]), r["B"] or -1, r["K"] or -1))]
    write_text(OUT/"ALL_MODELS.md", "# 全部224个已保存选择端点\n\n"
               "MLE每N只拟合一次，不因图中跨B重复显示而重复计数。\n\n"
               +table(["d", "stream", "method", "N", "B", "K", "(u,v)", "参考joint KL"], full_rows))
    write_text(OUT/"ALL_RISK_CONTRASTS.md", "# 原32项风险对比\n\n"+contrast_table(data["risk"]))
    write_text(OUT/"ALL_SCORE_CONTRASTS.md", "# 原24项独立确认评分对比\n\n"+contrast_table(data["scores"]))
    cells = []
    for d in (16, 64):
        for N in (256, 4096):
            for B in (16, 128):
                values = [next(r["mean_joint_KL"] for r in summaries
                               if (r["d"], r["N"], r["B"], r["method"]) == (d, N, B, method))
                          for method in METHODS]
                cells.append([d, N, B]+[number(v) for v in values])
    failed = []
    for family, records in (("risk32", data["risk"]), ("score24", data["scores"])):
        for r in records:
            if not r["precision_pass"]:
                failed.append([family, r["d"], r["N"], r["B"], r["comparison"],
                               number(r["mean"]), number(r["SE"]), number(r["SE_gate"]),
                               f'[{number(r["lower"])},{number(r["upper"])}]'])
    zero_rows = [r for r in data["risk"]+data["scores"] if r["SE"] == 0]
    primary = data["decision"]["primary_readouts"]
    primary_table = table(["d", "对比", "原门类型", "状态"],
                          [[r["d"], r["comparison"], r["kind"], "通过" if r["passed"] else "**未过**"]
                           for r in primary])
    failed_table = (table(["family", "d", "N", "B", "对比", "mean", "SE", "门", "区间"], failed)
                    if failed else "本次原风险／评分家族没有精度失败。")
    floor_table = table(["d", "floor选择/8", "概率下端", "固定参考风险期望下界", "原门"],
                        [[r["d"], r["floor_selections"], number(r["probability_lower"]),
                          number(r["expected_fixed_reference_risk_lower"]),
                          "通过" if r["passed"] else "未过"] for r in data["floor"]])
    selected_high = [r for r in data["risk"] if r["N"] == 4096 and r["B"] == 128]
    d16_noninfer = next(r for r in selected_high if r["d"] == 16 and r["comparison"] == "col_d_minus_MLE")
    high_budget_notes = table(["d", "高N高B对比", "均值", "原同时t区间", "解释"],
        [[r["d"], r["comparison"], number(r["mean"]),
          f'[{number(r["lower"])},{number(r["upper"])}]',
          "同K主要比较" if r["comparison"] == "raw_minus_col_d" else "追加K的辅助比较"]
         for r in selected_high if r["comparison"] in ("raw_minus_col_d", "col_d_minus_col_4d")])
    mean = data["mean"]
    text = f"""# 有限N与共享粒子集学习：完整可读结果

本页只汇总已完成并封存的RUN、REF和MEAN_DISPLAY，未新增样本、拟合或积分。raw与col共用标签及公共因子粒子前缀；raw另含私有Gaussian噪声，col将其边际化。已知正确组、零均值、正相关c=.9、真尺度(6,2)，两尺度都在相同49点网格内选择。

N为标签数，B为log外独立bank平均数，K为log内粒子数。N和B各自嵌套；每维8条独立流。主同K比较为raw K=d对col K=d，col K=4d是增加预算的辅助臂；数值MLE使用同一标签。没有将相同K称为相同计算成本。

## 1. 全部d/N/B风险均值

单位为完整向量的参考joint KL（nats），不是每坐标风险。MLE在每个N只选择一次，跨B仅重复展示同一输出。

{table(["d","N","B","raw K=d","col K=d","col K=4d","MLE"], cells)}

在N=4096、B=128两格，同K raw−col的原区间均支持超过.002的风险改善。d16的col K=d−MLE均值为{number(d16_noninfer["mean"])}，区间[{number(d16_noninfer["lower"])},{number(d16_noninfer["upper"])}]跨过.001非劣界，因此未通过；这不等于已证明col劣于该容许界，也不是已证明修复失败。col K=4d是额外预算，其相对K=d的原配对区间两维均跨0，不能把更好的点估计称作已确认的K收益。

{high_budget_notes}

N=256的四个raw−col比较均未过原精度门。N和B分别变化的所有点都保留；B从16到128的均值不是普遍单调，未另设事后B效应成功门。

![全部标签量与bank预算。点为8条流，线为均值；近零线性、其余对数刻度。未给方法均值新增区间。](../../../artifacts/finite_learning/figures/risk_by_N_B.png)

## 2. 原主门、全家族与失败

原主读数{sum(r["passed"] for r in primary)}/{len(primary)}通过；风险精度{data["decision"]["risk_precision_pass"]}/32，评分精度{data["decision"]["score_precision_pass"]}/24。非劣门不是双侧等效证明，失败项不借辅助K或不同N/B追认。

{primary_table}

![原32项配对风险对比，红叉保留精度失败。](../../../artifacts/finite_learning/figures/risk_contrasts.png)

{failed_table}

有{len(zero_rows)}条原对比的SE为0；这可由8个离散选点风险相同造成，不认证程序总体零不确定性或严格等效。风险32项与评分24项分别保留alpha=.025的原t近似家族；不重新按图中子集缩窄区间。

[全部224端点](../../../artifacts/finite_learning/figures/ALL_MODELS.md) · [全部32风险对比](../../../artifacts/finite_learning/figures/ALL_RISK_CONTRASTS.md) · [全部24独立评分对比](../../../artifacts/finite_learning/figures/ALL_SCORE_CONTRASTS.md)。

## 3. 预定floor概率读数与事件

{floor_table}

上述floor概率读数使用另一个预定二项家族；参考风险表下界不等于严格认证真实Gaussian积分误差，也不与t家族合称整个研究的一个95%保证。

![高N高B的三个完整PMF事件概率。](../../../artifacts/finite_learning/figures/event_probabilities.png)

三个事件为中央|sum Y|<=sqrt(d)、上尾和下尾，均从完整双count参考PMF求和，没有新增预测抽样。RUN全部模型固定零均值。

## 4. 均值补偿图的正确范围

![冻结的normal-limit曲线重绘，图例移到数据区域外。](../../../artifacts/finite_learning/figures/mean_compensation.png)

该图是normal-limit展示，不是本RUN的finite-d拟合，也没有重新求根。冻结匹配均值漂移为{number(mean["matched_beta"])}，中央概率{number(mean["candidate_center"])}与teacher相同；左右尾分别{number(mean["candidate_left"])}、{number(mean["candidate_right"])}，teacher每尾{number(mean["teacher_each_tail"])}。两个尾的平均binary Brier超额为{number(mean["pair_brier"])}，不是三类别Brier。旧MEAN_DISPLAY数据、图和冻结都未修改。

## 5. 数值与资源口径

参考以GH节点收敛、独立quad和完整PMF变体检查合格，不是严格积分余项证书；exact-node插值误差界不包含GH节点误差。详细见[REF_RESULTS](REF_RESULTS.md)。

原RUN资源计数：新训练teacher向量{data["counts"]["new_training_teacher_vectors"]}，独立确认teacher向量{data["counts"]["new_confirmation_teacher_vectors"]}，最终端点{data["counts"]["final_selection_endpoints"]}，0新神经网络、0新增部署抽样。候选、前缀和重复显示MLE不增加独立统计单位。完整原COUNTS.json及所有源hash由图表清单绑定。

这些图表不把有限网格程序称为连续总体全局最优，也不将小均值参数误差替代联合风险。图的结论以实际原主门和全部失败为准。
"""
    write_text(DOC/"RESULTS.md", text)


def build():
    assert not (OUT/"MANIFEST.json").exists(), "figure package already sealed"
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "finite-learning-fixed-results", "savefig.pad_inches": .1})
    inputs = {rel(p): sha(p) for p in source_paths()}
    risk_figure(data)
    contrast_figure(data)
    event_figure(data)
    mean_figure(data)
    readable(data)
    verify(inputs)
    write_json(OUT/"SOURCE_HASHES.json", inputs)
    write_json(OUT/"BUILD.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "new_fits": 0, "new_samples": 0,
        "new_integrals": 0, "mean_display_root_reexecuted": False,
        "old_scientific_files_modified": False, "visual_QA_pending": True,
        "figures": list(FIGURE_NAMES), "model_endpoints": 224,
        "risk_contrasts": 32, "score_contrasts": 24})
    print("Built unsealed figures/readable tables; visual QA required before seal.")


def seal(visual_qa_complete):
    assert visual_qa_complete, "explicit visual QA completion flag required"
    assert not (OUT/"MANIFEST.json").exists()
    assert not read(RUN/"DECISION.json")["continuous_global_optimality_certified"]
    sources = read(OUT/"SOURCE_HASHES.json")
    verify(sources)
    audit_paths = [REF/"INDEPENDENT_AUDIT.json", RUN/"INDEPENDENT_AUDIT.json"]
    for path in audit_paths:
        assert read(path)["passed"], str(path)
    build_record = read(OUT/"BUILD.json")
    build_record["visual_QA_pending"] = False
    build_record["visual_QA_completed_utc"] = datetime.now(timezone.utc).isoformat()
    build_record["visual_QA_scope"] = "all four PNGs inspected; legends outside; zero-risk and failure rows retained"
    write_json(OUT/"BUILD.json", build_record)
    outputs = [OUT/f"{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("png", "svg")]
    outputs += [OUT/name for name in ("SUMMARY.json", "ALL_MODELS.md", "ALL_RISK_CONTRASTS.md",
                                      "ALL_SCORE_CONTRASTS.md", "SOURCE_HASHES.json", "BUILD.json")]
    outputs += [DOC/"RESULTS.md"]
    assert all(p.is_file() for p in outputs)
    write_json(OUT/"MANIFEST.json", {
        "sealed_utc": datetime.now(timezone.utc).isoformat(), "source_files": sources,
        "output_files": {rel(p): sha(p) for p in outputs},
        "new_fits": 0, "new_samples": 0, "new_integrals": 0,
        "visual_QA_completed": True, "matplotlib_version": matplotlib.__version__,
        "independent_audits": {rel(p): sha(p) for p in audit_paths},
        "MEAN_DISPLAY_frozen_sources_unchanged": True,
        "mean_display_is_normal_limit_not_finite_fit": True})
    print("Finite-learning figures sealed.")


def check():
    manifest = read(OUT/"MANIFEST.json")
    verify(manifest["source_files"])
    verify(manifest["output_files"])
    verify(manifest["independent_audits"])
    assert manifest["visual_QA_completed"]
    print("Finite-learning figure sources/output hashes valid; no science recomputation.")


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
