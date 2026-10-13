"""Read-only science inputs -> static figures and tables; no fits/RNG/integration."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
from collections import Counter
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
BASE = ROOT/"artifacts/objective_strengthening"
OUT = BASE/"figures"
DOC = ROOT/"docs/08_research_研究工作/总体目标塌缩与同模型修复_20261001"
BLUE, ORANGE, GREEN, RED, GRAY = "#0072B2", "#D55E00", "#009E73", "#B22222", "#626B73"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
    "grid.linewidth": .6, "savefig.facecolor": "white", "figure.facecolor": "white",
    "svg.fonttype": "path", "svg.hashsalt": "objective-strengthening-20261001",
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csvread(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()+"\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load():
    paths = [DOC/"THEORY.md", DOC/"THEORY_SOURCE_NOTE.md"]
    for package, source, result in [
        ("N", "FREEZE.json", "RESULTS_FREEZE.json"),
        ("R", "FREEZE.json", "RESULTS_FREEZE.json"),
        ("B", "SOURCE_SEAL.json", "RESULTS_SEAL.json")]:
        for name in (source, result):
            p = BASE/package/name
            seal = read(p)
            for original, digest in seal["files"].items():
                assert sha(ROOT/original) == digest, original
            paths.append(p)
    names = {
        "N": ["CONFIG.json", "RESULTS.json", "LIMITS.json", "COUNTS.json", "MOMENTS.json"],
        "R": ["CONFIG.json", "RESULTS.json", "EVENTS.json", "COUNTS.json"],
        "B": ["CONFIG.json", "DECISION.json", "COUNTS.json", "CONTRASTS_35.csv",
              "ENDPOINTS.csv", "POSSIBLE_GRID_OPTIMAL_SETS.json", "REFERENCE_RISK.csv"],
    }
    data = {}
    for package, files in names.items():
        data[package] = {}
        for name in files:
            path = BASE/package/name
            paths.append(path)
            data[package][name] = csvread(path) if path.suffix == ".csv" else read(path)
    return data, paths


def b_summary(data):
    b = data["B"]
    endpoints = b["ENDPOINTS.csv"]
    contrasts = b["CONTRASTS_35.csv"]
    by_name = {r["comparison"]: r for r in contrasts}
    rows = []
    for budget in b["CONFIG.json"]["training_banks"]:
        row = {"B": budget}
        for mode in ("population", "finite"):
            selected = [r for r in endpoints if int(r["B"]) == budget and r["mode"] == mode]
            assert len(selected) == 16
            choices = Counter((float(r["u"]), float(r["v"])) for r in selected)
            row[mode+"_merged"] = sum(float(r["u"]) == float(r["v"]) for r in selected)
            row[mode+"_choices"] = [{"scales": list(k), "count": v} for k, v in sorted(choices.items())]
            row[mode+"_mean_KL"] = float(np.mean([float(r["true_joint_KL"]) for r in selected]))
        row["confirm_F"] = by_name[f"population_selected_B{budget}_confirm_F_minus_truth"]
        row["population_KL"] = by_name[f"population_selected_B{budget}_true_KL"]
        row["finite_delta_KL"] = by_name[f"finite_selected_B{budget}_KL_minus_old_J"]
        row["region"] = next(r for r in b["POSSIBLE_GRID_OPTIMAL_SETS.json"] if r["B"] == budget)
        rows.append(row)
    return rows


def save(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=180)
    fig.savefig(OUT/(name+".svg"), metadata={"Date": None})
    svg = OUT/(name+".svg")
    write_text(svg, svg.read_text(encoding="utf-8"))
    plt.close(fig)


def plot_N(data):
    rows, limits = data["N"]["RESULTS.json"], data["N"]["LIMITS.json"]["limits"]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.5), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.2, top=.78, wspace=.12)
    for ax, sign, title in zip(axes, (-1, 1), ("Negative correlation", "Positive correlation")):
        values = sorted((r for r in rows if r["sign"] == sign), key=lambda r: r["d"])
        x = [r["d"] for r in values]
        limit = next(r for r in limits if r["sign"] == sign)
        for law, color, marker in ((0, BLUE, "o"), (1, ORANGE, "s")):
            y = np.array([r["probabilities"][law] for r in values])
            bounds = np.array([r["probability_intervals"][law] for r in values])
            ax.errorbar(x, y, yerr=np.array([y-bounds[:, 0], bounds[:, 1]-y]),
                        color=color, marker=marker, lw=1.5, ms=5.5, capsize=3)
            ax.axhline(limit["teacher_event" if law == 0 else "floor_event"],
                       color=color, ls=(0, (5, 3)), lw=1.2, alpha=.85)
        ax.plot(x, [r["independent_binomial_event"] for r in values],
                color=GRAY, marker="D", ms=4, lw=1.25)
        ax.axhline(limit["independent_event"], color=GRAY, ls=(0, (5, 3)), lw=1.1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        ax.set_xlabel("Dimension d (log scale)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(.56, .935)
        ax.set_xlim(13, 1280)
    axes[0].set_ylabel("Probability of the global count event")
    handles = [
        Line2D([], [], color=BLUE, marker="o", label="Teacher (6, 2)"),
        Line2D([], [], color=ORANGE, marker="s", label="Fixed floor (1, 1)"),
        Line2D([], [], color=GRAY, marker="D", label="Exact independent Binomial"),
        Line2D([], [], color="#333333", ls=(0, (5, 3)), label="CLT limit (matched color)"),
    ]
    fig.suptitle("N  |  A global count event distinguishes truth from the fixed floor",
                 y=.98, fontsize=14, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .91), ncol=4, frameon=False)
    fig.text(.5, .09, r"Event: $|\sum_iY_i|\leq\sqrt{d}$.  65,536 observations per law and cell; teacher/floor draws are coupled.",
             ha="center", fontsize=9)
    fig.text(.5, .045, "Error bars: original simultaneous 95% Clopper-Pearson family of 16 probabilities. Fixed baselines; no model fitting.",
             ha="center", fontsize=9)
    save(fig, "N_event_probabilities")


def plot_R(data):
    rows = data["R"]["RESULTS.json"]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.3))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.8, wspace=.3)
    ax = axes[0]
    positions = np.arange(len(rows))
    y = np.array([r["confirmation_truth_minus_selected"]["mean"]/r["d"] for r in rows])
    lo = np.array([r["confirmation_truth_minus_selected"]["interval"][0]/r["d"] for r in rows])
    hi = np.array([r["confirmation_truth_minus_selected"]["interval"][1]/r["d"] for r in rows])
    ax.errorbar(positions, y, yerr=[y-lo, hi-y], fmt="o", color=BLUE, capsize=4, ms=6)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.set_xticks(positions, [f"d={r['d']}\nK={r['K']}" for r in rows])
    ax.set_ylim(-.025, .86)
    ax.set_ylabel("Truth minus selected raw score / d\n(nats per coordinate; not joint KL)")
    ax.set_title("Independent score confirmation", fontweight="bold")
    for x, value in zip(positions, y):
        ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)
    # Collapse identical event evidence across K; do not draw it twice.
    unique = []
    for d in sorted({r["d"] for r in rows}):
        same_d = [r for r in rows if r["d"] == d]
        first = same_d[0]
        for other in same_d[1:]:
            assert other["mean_event_brier_plugin"] == first["mean_event_brier_plugin"]
            assert other["mean_event_brier_interval"] == first["mean_event_brier_interval"]
            assert other["models"] == first["models"]
        unique.append(first)
    ax = axes[1]
    x = np.arange(len(unique))
    y = np.array([r["mean_event_brier_plugin"] for r in unique])
    bounds = np.array([r["mean_event_brier_interval"] for r in unique])
    ax.errorbar(x, y, yerr=[y-bounds[:, 0], bounds[:, 1]-y],
                fmt="s", color=ORANGE, ms=6, capsize=4)
    ax.axhline(.001, color=GRAY, ls="--", lw=1.1)
    ax.text(2.38, .00115, "original gate", ha="right", fontsize=8.5, color=GRAY)
    ax.set_xticks(x, ["d=16\nK=10", "d=64\nK=10 and 100", "d=256\nK=10"])
    ax.set_xlim(-.45, 2.45)
    ax.set_ylim(0, .010)
    ax.set_ylabel("Event Brier excess\n(plug-in point; CP-rectangle interval)")
    ax.set_title("Three distinct event-risk estimates", fontweight="bold")
    ax.annotate("shared probability evidence", (1, y[1]),
                xytext=(1.32, .0028), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=GRAY), ha="center")
    fig.suptitle("R  |  Sealed finite-grid selections beat truth in raw-score confirmation",
                 y=.98, fontsize=13.6, fontweight="bold")
    fig.text(.5, .145, "All 32 procedures selected (1, 1); this is not a continuous population-optimum certificate.",
             ha="center", fontsize=9)
    fig.text(.5, .097, "Scores: 8 streams, family 4 t-approximation, alpha=0.025. Events: original family 38 CP, alpha=0.025.",
             ha="center", fontsize=9)
    fig.text(.5, .05, "d=64 shares labels/prior prefixes across K and reuses one event probability; event intervals condition on the sealed models.",
             ha="center", fontsize=9)
    save(fig, "R_raw_confirmation")


def plot_B(summary):
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 5.6))
    fig.subplots_adjust(left=.055, right=.985, bottom=.27, top=.8, wspace=.34)
    x = np.arange(3)
    ax = axes[0]
    for offset, key, color, label in [
        (-.1, "possible_optimal_KL", BLUE, "Possible grid optima"),
        (.1, "possible_near_optimal_KL", GREEN, "Possible .005-near optima")]:
        for position, row in zip(x+offset, summary):
            values = row["region"][key]
            ax.vlines(position, values["minimum_KL"], values["maximum_KL"], color=color, lw=5)
            ax.scatter([position, position], [values["minimum_KL"], values["maximum_KL"]],
                       marker="_", s=90, color=color)
            ax.text(position, values["maximum_KL"]+.002, str(values["count"]),
                    ha="center", va="bottom", fontsize=8.5, color=color)
        ax.plot([], [], color=color, lw=4, label=label)
    ax.set_ylim(0, .079)
    ax.set_ylabel("True joint KL range across retained points")
    ax.set_title("KL ranges across retained regions", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.2, frameon=False)
    ax = axes[1]
    pop = [r["population_merged"] for r in summary]
    finite = [r["finite_merged"] for r in summary]
    ax.bar(x-.17, pop, width=.32, color=BLUE, label="Population-label weighting")
    ax.bar(x+.17, finite, width=.32, color=ORANGE, label="Reused finite labels")
    for positions, values in ((x-.17, pop), (x+.17, finite)):
        for position, value in zip(positions, values):
            ax.text(position, value+.28, str(value), ha="center", fontsize=9)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylabel("Equal-scale selections out of 16")
    ax.set_title("Merging persists as B increases", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.1, frameon=False)
    ax = axes[2]
    for position, row in zip(x, summary):
        value = row["finite_delta_KL"]
        mean, low, high = [float(value[k]) for k in ("mean", "lower", "upper")]
        valid = value["precision_pass"] == "True"
        color = BLUE if valid else RED
        ax.errorbar(position, mean, yerr=[[mean-low], [high-mean]],
                    fmt="o" if valid else "D", ms=6, color=color,
                    markerfacecolor=color if valid else "white", capsize=4)
    ax.axhline(0, color=GRAY, lw=.9)
    ax.set_ylim(-.001, .028)
    ax.set_ylabel("True joint KL: selected G minus old J")
    ax.set_title("Secondary finite-label comparison", fontweight="bold")
    ax.annotate("SE gate failed\n0.001016 > 0.001", xy=(0, .019081535),
                xytext=(.35, .0255), fontsize=8.2, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED), ha="left")
    for ax in axes:
        ax.set_xticks(x, ["64", "256", "1024"])
        ax.set_xlabel(r"$B_{\rm MC}$ (K remains 10)")
        ax.set_xlim(-.45, 2.45)
    fig.suptitle("B  |  More banks refine the same finite-K group objective",
                 y=.98, fontsize=14, fontweight="bold")
    fig.text(.5, .17, "Left: numbers count grid points; bars show KL min-max ranges, not risk CIs. d=16, fixed 361-point grid.",
             ha="center", fontsize=9)
    fig.text(.5, .12, "Bands: family 1080, alpha=0.025. Contrasts: family 35, alpha=0.025. t approximations, 16 streams, qualified old references.",
             ha="center", fontsize=9)
    fig.text(.5, .07, "At B=1024 all population selections coincide: their degenerate KL interval is not zero uncertainty about future selections.",
             ha="center", fontsize=9)
    fig.text(.5, .025, "This group-score experiment does not test the new raw fixed-floor theorem; (1,1) is an anchor, not this grid's lower boundary.",
             ha="center", fontsize=8.7)
    save(fig, "B_bank_regions")


def fmt(value, digits=6):
    return f"{float(value):.{digits}f}"


def ci(interval, digits=6):
    return "["+", ".join(fmt(x, digits) for x in interval)+"]"


def contrast_ci(row, digits=6):
    return ci([float(row["lower"]), float(row["upper"])], digits)


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join("---" for _ in headers)+" |"]+
                     ["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def display_choices(values):
    return "，".join(f"{r['count']}×({r['scales'][0]:g},{r['scales'][1]:g})" for r in values)


def report(data, budgets):
    N, R, B = data["N"], data["R"], data["B"]
    parts = ["# 总体目标塌缩与同模型修复：可读结果\n",
        "本页只汇总已完成的 T 理论与已封存 N/B/R 数据，不新增拟合、随机数、积分或置信家族。三包统计合同分别保留，不能拼成一次全研究95%保证。表格仅作显示舍入，判定均读取未舍入原值；所有图使用保存值。图表与来源哈希见 [figures/MANIFEST.json](../../../artifacts/objective_strengthening/figures/MANIFEST.json)。\n",
        "## 1. 三个经验问题与理论接口\n",
        table(["包", "实际回答", "不能据此声称"], [
            ["T", "固定正下限 raw 总体近最优点的尺度定位、两符号事件后果；正族 private-RB 的同模型风险上界", "原自由 Cholesky/SGD/任意有限维精确贴边；col有限拟合已完成"],
            ["N", "真模型与固定(1,1)基线的有限维事件差", "已拟合端点或有限维总体极小点"],
            ["R", "负族有限网格 raw 选点经独立分数及事件确认", "连续总体全局最优或优化轨迹认证"],
            ["B", "固定K10 group目标增加外层bank数后是否仍偏好错误区域", "增加B等于增加K，或直接验证新raw定理"]]),
        "\n理论条件见 [THEORY.md](THEORY.md)。R/B 都没有运行正族 private-RB 的新有限数据拟合；理论修复与这次有限实验须分开。\n",
        "## 2. N：固定全局事件，正负两族完整读数\n",
        "事件为 $|\\sum_iY_i|\\le\\sqrt d$。每格65536个独立base向量，teacher与floor在格内共享Gaussian与Uniform；16个概率使用原Bonferroni同时95%双侧Clopper–Pearson区间。下列Brier点值是概率差平方的plug-in，不称无偏风险估计；区间来自两概率矩形。\n"]
    nrows = []
    for r in N["RESULTS.json"]:
        p, q = r["probabilities"]
        nrows.append([r["d"], "负" if r["sign"] < 0 else "正",
            f"{fmt(p)} {ci(r['probability_intervals'][0])}",
            f"{fmt(q)} {ci(r['probability_intervals'][1])}",
            fmt(r["independent_binomial_event"]), ci(r["risk_rectangle"]["difference"]),
            f"{fmt((p-q)**2)} {ci(r['risk_rectangle']['brier'])}",
            ci(r["risk_rectangle"]["logloss"]), "通过" if r["primary_pass"] else "未过"])
    parts.append(table(["d", "符号", "teacher概率 [CP]", "floor概率 [CP]", "独立Binomial",
                        "teacher−floor区间", "Brier点值 [范围]", "事件log-loss范围", "原主门"], nrows))
    parts += ["\n原主门为方向区间不含0且Brier下端>.001，8/8通过；不是对新拟合器的8个成功端点。\n",
              "![N finite event probabilities](../../../artifacts/objective_strengthening/figures/N_event_probabilities.png)\n",
              "### 解析极限与空泛粗界\n"]
    parts.append(table(["符号", "teacher极限", "floor极限", "独立极限", "teacher−floor", "Brier极限", "事件log-loss极限"],
        [["负" if r["sign"] < 0 else "正", fmt(r["teacher_event"]), fmt(r["floor_event"]),
          fmt(r["independent_event"]), fmt(r["teacher_minus_floor"]),
          fmt(r["brier_excess"]), fmt(r["event_logloss_excess"])] for r in N["LIMITS.json"]["limits"]]))
    limits, moment = N["LIMITS.json"], N["MOMENTS.json"]
    parts += [f"\n这些极限来自已保存GH最后阶数；GH阶间差 {moment['last_order_difference']:.3g}、独立quad差 {moment['independent_difference']:.3g}，均过原1e−10门。它们是收敛核对，不是严格积分区间。\n",
              table(["固定诊断", "粗尺度偏离和上界", "区间给出的平凡最大值", "结论"],
                    [["负族 d128/K100/[1,8]", fmt(limits["coarse_bound_d128_K100"]), fmt(limits["trivial_max_scale_excess"]), "20.845 > 14，有限维定位界无信息"]]),
              "\n该粗界采用负族的较紧谱上界，不是正族谱界；不画成有限维sharp定位曲线。旧独立候选的事件差也不与当前固定a=1基线混用。\n",
              "## 3. R：有限网格 raw 选择与独立确认\n",
              "候选[1,8]²、正确已知组、负相关c=.9、真(6,2)。32个粗/局部细网格程序全部选(1,1)。四项分数差为truth减已封存选点，独立确认8流，df7、原family4、alpha=.025的t近似；精度门SE/d<=.002，方向门下端/d>.001。\n"]
    rtable = []
    for r in R["RESULTS.json"]:
        f = r["confirmation_truth_minus_selected"]
        rtable.append([r["d"], r["K"], f"{r['floor_choices']}/8", fmt(f["mean"]),
                       ci(f["interval"]), fmt(f["mean"]/r["d"]),
                       f"{fmt(r['mean_event_brier_plugin'])} {ci(r['mean_event_brier_interval'])}",
                       "通过" if r["primary_score_pass"] else "未过",
                       "通过" if r["primary_event_pass"] else "未过"])
    parts.append(table(["d", "K", "选(1,1)", "确认ΔF总nats", "原t区间", "ΔF/d", "事件Brier [范围]", "分数门", "事件门"], rtable))
    parts += ["\n分数是原始粒子NLL差，不是joint KL。4/4分数与4/4预定事件门通过；但d64两个K共享标签/prior前缀，选中相同模型并共用同一事件概率，不能称第四份独立预测验证。\n",
              "事件采用新观察，与N包独立。实际只有6个不同(d,u,v)状态，仍保留预先冻结的38概率family、alpha=.025 CP合同；不事后缩为6。下列是全部状态：\n"]
    parts.append(table(["d", "(u,v)", "事件成功数/65536", "概率", "原CP区间"],
        [[r["state"][0], f"({r['state'][1]:g},{r['state'][2]:g})", f"{r['event_count']}/{r['N']}",
          fmt(r["probability"]), ci(r["interval"])] for r in R["EVENTS.json"]]))
    parts += ["\n事件范围条件于已封存的实际模型。各格8个已实现模型的平均区间，不是训练算法未来期望风险的严格区间；同一模型重复出现不增加事件重复数。没有连续最优或SGD保证。\n",
              "![R confirmation](../../../artifacts/objective_strengthening/figures/R_raw_confirmation.png)\n",
              "## 4. B：固定K=10增加独立bank平均数\n",
              "d16、真(6,2)、正确已知组，361点网格仍含0与.5，(1,1)只是锚而不是可行域floor。人口层以旧合格真实81格概率加权，排除有限标签噪声；有限bank误差仍存在。16条独立流的B=64/256/1024为嵌套前缀，全部选点封存后，每流4096个新confirmation banks复评。B始终在log外平均，没有把十粒子改变成更多粒子。\n",
              "### 人口加权选点与独立确认\n"]
    parts.append(table(["B_MC", "16流选点", "合并数", "真joint KL均值 [原t区间]", "确认F(选点)−F(truth) [原t区间]"],
        [[r["B"], display_choices(r["population_choices"]), f"{r['population_merged']}/16",
          f"{fmt(r['population_mean_KL'],8)} {contrast_ci(r['population_KL'],8)}",
          f"{fmt(r['confirm_F']['mean'],8)} {contrast_ci(r['confirm_F'],8)}"] for r in budgets]))
    parts += ["\nB1024的16流全选同一个离散点，其KL区间退化为[0.01754866,0.01754866]。保留原规则的显示值和资格，但它不是严格零不确定性、不是未来选点概率为1或总体等效证明。\n",
              "### 不依赖赢家的固定锚\n"]
    fixed = [r for r in B["CONTRASTS_35.csv"] if r["kind"] == "fixed_confirmation_F"
             and (float(r["u"]), float(r["v"])) in {(3., 1.), (1., 1.), (1.5, 1.5)}]
    parts.append(table(["参数", "确认F(theta)−F(truth)", "原同时t区间", "SE", "原精度"],
        [[f"({float(r['u']):g},{float(r['v']):g})", fmt(r["mean"],8), contrast_ci(r,8),
          fmt(r["SE"],9), "通过" if r["precision_pass"] == "True" else "失败"] for r in fixed]))
    parts += ["\n固定(3,1)见证、高B人口选点确认评分优势、高B选点真实KL>.001三项原定主要读数均通过。全部35行在[完整对比表](../../../artifacts/objective_strengthening/figures/B_CONTRASTS_35.md)，没有为上表重建较小置信家族。\n",
              "### 可能网格最优及.005近优区域\n"]
    parts.append(table(["B_MC", "可能网格最优数", "这些点的joint KL范围", "可能.005近优数", "这些点的joint KL范围", "truth排除"],
        [[r["B"], r["region"]["possible_optimal_KL"]["count"],
          ci([r["region"]["possible_optimal_KL"][k] for k in ("minimum_KL","maximum_KL")],8),
          r["region"]["possible_near_optimal_KL"]["count"],
          ci([r["region"]["possible_near_optimal_KL"][k] for k in ("minimum_KL","maximum_KL")],8),
          "是" if r["region"]["truth_excluded"] else "否"] for r in budgets]))
    parts += ["\n集合沿原同时带规则L_i<=min U_j（近优再加.005）构造，条件于旧数值参考资格和t近似；条形是集合内真实KL的min–max范围，不是单个风险的置信区间，也不是连续参数全局证书。每流经验.005近优集已另存，其本身没有同时覆盖含义。\n",
              "### 复用旧65536标签的次要层及唯一SE失败\n"]
    parts.append(table(["B_MC", "有限标签16流选点", "合并数", "真joint KL均值", "G−旧J风险差", "原t区间", "SE", "精度"],
        [[r["B"], display_choices(r["finite_choices"]), f"{r['finite_merged']}/16",
          fmt(r["finite_mean_KL"],8), fmt(r["finite_delta_KL"]["mean"],8),
          contrast_ci(r["finite_delta_KL"],8), fmt(r["finite_delta_KL"]["SE"],10),
          "通过" if r["finite_delta_KL"]["precision_pass"] == "True" else "**失败**"] for r in budgets]))
    parts += ["\nB64有限标签G−J的SE=0.0010160151超过原.001门；即便所显示区间为正，该行仍未合格。35项中34项精度通过，1080条带全部通过。没有增加bank、seed或换门补救；有限旧标签层是次要开发证据。\n",
              "![B bank regions](../../../artifacts/objective_strengthening/figures/B_bank_regions.png)\n",
              "35项确认/风险家族使用alpha=.025，1080条population带另用alpha=.025，均为df15的t近似；覆盖解释还条件于旧参考数值资格。它们不构成严格积分余项或全流程连续最优认证。\n",
              "## 5. 资源与证据等级\n"]
    nc, rc, bc = N["COUNTS.json"], R["COUNTS.json"], B["COUNTS.json"]
    parts.append(table(["项目", "N", "R", "B"], [
        ["新观察用途", f"{nc['base_vector_draws']:,} base；{nc['coupled_law_observations']:,}耦合模型观察",
         f"{rc['new_teacher_vectors']:,}选择/确认teacher向量＋{rc['event_observations']:,}独立事件模型观察", "0新标签"],
        ["新不同潜粒子", "仅事件生成，不是likelihood训练bank", f"{rc['unique_prior_particles']:,}", f"{bc['new_scientific_prior_particles']:,}"],
        ["选点程序", "0", f"{rc['fitted_procedures']}最终程序；另{rc['coarse_intermediate_selections']}粗中间点", f"{bc['grid_selection_endpoints']}网格端点"],
        ["旧数据复用", "无", "d64两K共享前缀，不重复记标签", f"{bc['old_label_vectors_reused']:,}旧标签；{bc['reused_exact_J_fits']}旧J端点"],
        ["新神经网络", "0", "0", "0"],
    ]))
    parts += ["\nT为解析证明。N求积只经数值收敛门；N事件区间有其CP合同。R分数使用t近似且事件条件于所选模型。B使用t近似及旧参考；本包保留一个SE失败和离散选点造成的退化KL区间。三者不能互相追认门或混为同预算方法排名。\n",
              "以上图表是0新fit、0新RNG、0新积分的派生产物；不导入生产训练函数、不打开本地prior/标签数组。公开源和原封存文件的哈希先核对，再记录本图表自己的来源/产物清单。"]
    write_text(DOC/"RESULTS.md", "\n".join(parts))
    all_rows = [[r["comparison"], r["kind"],
                 f"({float(r['u']):g},{float(r['v']):g})" if r["u"] else "所选程序",
                 r["B"], fmt(r["mean"],9), fmt(r["SE"],10),
                 contrast_ci(r,9), r["SE_gate"], "通过" if r["precision_pass"] == "True" else "**失败**"]
                for r in B["CONTRASTS_35.csv"]]
    write_text(OUT/"B_CONTRASTS_35.md",
        "# B 全部35项原确认/风险对比\n\nfamily=35，alpha=.025，df15；t近似，未缩小家族。均保留原精度门。\n\n"+
        table(["comparison", "kind", "参数(u,v)", "B", "mean", "SE", "interval", "SE gate", "precision"], all_rows))


def build():
    assert not (OUT/"MANIFEST.json").exists(), "sealed figures cannot be overwritten"
    OUT.mkdir(parents=True, exist_ok=True)
    data, paths = load()
    budgets = b_summary(data)
    plot_N(data)
    plot_R(data)
    plot_B(budgets)
    report(data, budgets)
    paths.append(Path(__file__).resolve())
    write_json(OUT/"SOURCE_HASHES.json", {p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    print("Three figure pairs and RESULTS.md generated; visual QA before seal", flush=True)


def seal():
    path = OUT/"MANIFEST.json"
    assert not path.exists()
    inputs = read(OUT/"SOURCE_HASHES.json")
    for name, digest in inputs.items():
        assert sha(ROOT/name) == digest, name
    outputs = sorted(OUT.glob("*.png"))+sorted(OUT.glob("*.svg"))+[OUT/"B_CONTRASTS_35.md", OUT/"SOURCE_HASHES.json", DOC/"RESULTS.md"]
    assert len(list(OUT.glob("*.png"))) == len(list(OUT.glob("*.svg"))) == 3
    value = dict(sealed_utc=datetime.now(timezone.utc).isoformat(), source_files=inputs,
        output_files={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
        matplotlib_version=matplotlib.__version__, new_fits=0, new_random_draws=0, new_integrals=0,
        original_scientific_results_modified=False, original_confidence_families_retained=True,
        R_d64_event_evidence_displayed_once=True,
        visual_QA="N/R/B PNGs visually inspected before seal; B range note moved outside data region",
        visual_QA_required_before_this_seal=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Figure manifest sealed", sha(path), flush=True)


def check():
    manifest = read(OUT/"MANIFEST.json")
    for mapping in (manifest["source_files"], manifest["output_files"]):
        for name, digest in mapping.items():
            assert sha(ROOT/name) == digest, name
    print("Figure sources/output hashes valid; no science recomputation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "seal", "check"])
    args = parser.parse_args()
    {"build": build, "seal": seal, "check": check}[args.phase]()
