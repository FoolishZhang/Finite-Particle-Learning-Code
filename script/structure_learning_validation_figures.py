"""Read frozen D tables and produce separate validation figures; no new statistics."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/structure_learning_v2"
D = BASE / "D"
OUT = BASE / "validation_figures"
DOC = ROOT / "docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930/D_RESULTS.md"
SOURCE_NAMES = ["FREEZE.json", "CANDIDATES_FREEZE.json", "RESULTS_FREEZE.json",
                "MANIFEST.json", "VALIDATION_RESULTS.csv", "SUMMARY.csv", "DECISION.json"]
OUTPUT_NAMES = ["eb_intervals.png", "eb_intervals.svg", "joint_risk.png", "joint_risk.svg"]
COLORS = {"GP": "#177f73", "H0": "#66758d"}
RISK_COLORS = ["#1775a5", "#ce812a", "#178676"]
DISPLAY_ZERO = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csvread(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_guard():
    for name in ("FREEZE.json", "CANDIDATES_FREEZE.json", "RESULTS_FREEZE.json", "MANIFEST.json"):
        for path, expected in read(D / name)["files"].items():
            assert sha(ROOT / path) == expected, path
    assert read(D / "STATE.json")["status"] == "complete"
    rows = csvread(D / "VALIDATION_RESULTS.csv")
    summary = csvread(D / "SUMMARY.csv")
    assert len(rows) == 32 and len(summary) == 2
    assert {(r["profile"], int(r["stream"])) for r in rows} == {
        (p, s) for p in ("null", "contrast") for s in range(16)}
    assert all(int(row["n"]) == 262144 for row in rows)
    return rows, {r["profile"]: r for r in summary}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8fa0b4", "xtick.color": "#334155", "ytick.color": "#334155",
        "text.color": "#172338", "axes.labelcolor": "#25364d",
        "grid.color": "#dce4ec", "grid.linewidth": .65,
        "legend.frameon": False, "savefig.facecolor": "white",
        "figure.facecolor": "white", "svg.fonttype": "path",
        "svg.hashsalt": "structure_learning_validation_D",
    })


def normalize_text(path):
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()+"\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def save(fig, stem):
    fig.savefig(OUT / (stem+".png"), dpi=180, bbox_inches="tight", pad_inches=.18)
    fig.savefig(OUT / (stem+".svg"), bbox_inches="tight", pad_inches=.18,
                metadata={"Date": None, "Creator": "structure_learning_validation_figures.py"})
    plt.close(fig)
    normalize_text(OUT / (stem+".svg"))


def eb_plot(rows, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.0))
    fig.subplots_adjust(left=.075, right=.975, top=.775, bottom=.185, wspace=.28)
    fig.suptitle("Independent validation for candidate selection",
                 fontsize=17, weight="bold", y=.975)
    fig.text(.5, .928, "32 two-sided simultaneous 95% empirical Bernstein intervals",
             ha="center", fontsize=11, color="#41516a")
    fig.text(.5, .890, "Fixed training candidates | 262,144 validation observations per case",
             ha="center", fontsize=10.5, color="#41516a")
    fig.legend(handles=[
        Line2D([0], [0], color=COLORS["H0"], marker="o", label="Select the fitted H0 candidate"),
        Line2D([0], [0], color=COLORS["GP"], marker="o", label="Retain the trained GP candidate")],
        loc="upper center", bbox_to_anchor=(.5, .859), ncol=2, columnspacing=3.5)
    for panel, (ax, profile) in enumerate(zip(axes, ("null", "contrast"))):
        data = sorted((row for row in rows if row["profile"] == profile), key=lambda r: int(r["stream"]))
        for row in data:
            y = int(row["stream"])
            mu, lo, hi = [1e6*float(row[key]) for key in ("mean", "lower", "upper")]
            color = COLORS[row["selected_model"]]
            ax.errorbar(mu, y, xerr=[[mu-lo], [hi-mu]], color=color,
                        marker="o", markersize=5.1, capsize=3.2,
                        linewidth=1.5, elinewidth=1.3, zorder=3)
            xtext = .53 if profile == "null" else 14.9
            ax.text(xtext, y, row["selected_model"], ha="center", va="center",
                    color=color, fontsize=9, weight="bold")
        ax.axvline(0, color="#20364f", linestyle="--", linewidth=1.2, zorder=2)
        ax.set_ylim(15.8, -1.0)
        ax.set_yticks(np.arange(16))
        ax.set_yticklabels([f"{s:02}" for s in range(16)])
        ax.set_ylabel("Stream")
        ax.set_xlabel(r"Pair-risk gain: loss(H0) - loss(GP)  ($10^{-6}$ nats/pair)")
        ax.grid(axis="y", alpha=.55)
        ax.grid(axis="x", alpha=.70)
        ax.set_axisbelow(True)
        ax.set_title(f"{chr(65+panel)}  |  {profile}: GP retained {summary[profile]['keep_GP']}/16",
                     loc="left", pad=12)
        if profile == "null":
            ax.set_xlim(-1.8, .77)
            ax.set_xticks([-1.5, -1, -.5, 0, .5])
            ax.text(.53, -.65, "Choice", ha="center", fontsize=9, color="#526176")
        else:
            ax.set_xlim(-.8, 15.7)
            ax.set_xticks([0, 5, 10, 15])
            ax.text(14.9, -.65, "Choice", ha="center", fontsize=9, color="#526176")
    fig.text(.075, .102, "Decision: retain GP only when the lower endpoint is strictly above zero; otherwise select H0.",
             fontsize=10, color="#334155")
    fig.text(.075, .060, "H0 is one training-fitted candidate. This is not a test of the entire homogeneous model class.",
             fontsize=10, color="#334155")
    save(fig, "eb_intervals")


def display_values(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < DISPLAY_ZERO, 0., values)


def risk_plot(rows, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.735, bottom=.25, wspace=.22)
    fig.suptitle("Joint KL after candidate selection", y=.975, fontsize=17, weight="bold")
    fig.text(.5, .919, "Evaluation only: joint KL did not enter the validation selector",
             ha="center", fontsize=11, color="#41516a")
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="#8794a8", linestyle="none", label="Each saved case"),
        Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor="#1d2d45",
               linestyle="none", color="#1d2d45", label="Existing summary mean")],
        loc="upper center", bbox_to_anchor=(.5, .88), ncol=2, columnspacing=3.0)
    fields = ["GP_joint_KL", "H0_joint_KL", "selected_joint_KL"]
    mean_fields = ["mean_GP_joint_KL", "mean_H0_joint_KL", "mean_selected_joint_KL"]
    jitter = np.linspace(-.15, .15, 16)
    for panel, (ax, profile) in enumerate(zip(axes, ("null", "contrast"))):
        data = sorted((row for row in rows if row["profile"] == profile), key=lambda r: int(r["stream"]))
        for method, (field, meanfield) in enumerate(zip(fields, mean_fields)):
            raw = [float(r[field]) for r in data]
            ax.scatter(method+jitter, display_values(raw), color=RISK_COLORS[method],
                       s=24, alpha=.85, zorder=3, linewidths=.4, edgecolors="white")
            mean = float(summary[profile][meanfield])
            shown = float(display_values([mean])[0])
            ax.scatter([method], [shown], marker="D", s=72, facecolors="white",
                       edgecolors="#152a42", linewidths=1.4, zorder=4)
            label = r"$\approx 0$" if abs(mean) < DISPLAY_ZERO else f"{mean:.3g}"
            below = profile == "contrast" and method in (0, 2)
            ax.annotate(label, xy=(method, shown), xytext=(0, -17 if below else 12),
                        textcoords="offset points", ha="center",
                        va="top" if below else "bottom", fontsize=9.5, color="#243751")
        ax.set_title(f"{chr(65+panel)}  |  {profile}", loc="left", pad=11)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Trained GP", "Fitted H0", "Selected"])
        ax.set_xlim(-.42, 2.42)
        ax.set_yscale("symlog", linthresh=1e-6, linscale=.75)
        ax.set_ylim(-1.4e-7, 6e-3)
        ax.set_yticks([0, 1e-6, 1e-5, 1e-4, 1e-3])
        ax.set_yticklabels([r"$\approx 0$", r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"])
        ax.grid(axis="y", alpha=.75)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Full-pattern joint KL  (nats; log-like axis)")
    fig.text(.09, .152, r"Display only: values with |KL| < $10^{-12}$ are drawn at zero ($\approx 0$). Raw values are retained.",
             fontsize=10, color="#334155")
    fig.text(.09, .097, "Dots show all 16 cases per profile; diamonds copy the frozen summary means. No new joint-KL intervals.",
             fontsize=10, color="#334155")
    fig.text(.09, .052, "Pair-risk evidence does not certify joint-KL improvement or true heterogeneity.",
             fontsize=10, color="#334155")
    save(fig, "joint_risk")


def results_document(rows, summary):
    csvlink = "../../../artifacts/structure_learning_v2/D/VALIDATION_RESULTS.csv"
    baselink = "../../../artifacts/structure_learning_v2/validation_figures"
    lines = [
        "# D：独立验证的候选选择", "",
        "本页仅整理已经封存的D结果，不修改其READABLE、CSV、候选或区间。32对模型都在验证前确定：复用C的32个GLOBAL_P端点，并在各自训练数据上拟合32个同质H0端点。每例使用262,144个独立验证观察，总计8,388,608；没有新GP拟合、参考积分或先验bank。", "",
        "固定规则只在EB区间下端严格大于0时保留GP。32项双侧同时95%区间比较的是本次训练得到的两个候选的pair风险；不构成整个同质模型类的检验，不认证真实异质性或完整joint KL改善。完整风险只在选择后查表评价。", "",
        f"![全部32个经验Bernstein区间]({baselink}/eb_intervals.png)", "",
        "图中两个面板的横轴范围不同，单位同为10⁻⁶ nats/pair。全部32条区间及其选择完整展示，0为唯一选择门。null没有正下界；部分null区间跨0，因此不能说全部GP的pair风险已被证明更差。", "",
        "|profile|保留GP|回退H0|原GP平均joint KL|H0平均joint KL（原值）|选择后平均joint KL（原值）|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in ("null", "contrast"):
        r = summary[profile]
        lines.append(f"|{profile}|{r['keep_GP']}/16|{r['choose_H0']}/16|"
            f"{float(r['mean_GP_joint_KL']):.12g}|{r['mean_H0_joint_KL']}|{r['mean_selected_joint_KL']}|")
    lines += [
        "", "null选择后的均值约为0；原始−2.3314683517128287×10⁻¹⁵是风险库的浮点舍入量，未改写或裁剪原值。contrast保留GP后的平均joint KL为6.86834081276011×10⁻⁶；H0均值为.0019421735751805613。它们是这32例的事后描述，不是新的joint风险置信区间或普遍方法保证。", "",
        f"![候选选择后的完整joint KL]({baselink}/joint_risk.png)", "",
        "仅为绘图，绝对值小于10⁻¹²的KL点显示在标为≈0的位置；这不是新的科学阈值。所有原值继续保留于下表和原CSV。点为全部逐例风险，菱形直接取已冻结SUMMARY中的均值；未拟合新模型或重算新统计家族。", "",
        "## 全部32例", "",
        "下表EB均值和区间使用nats/pair；joint KL使用完整16维模式nats，两种单位不能混作同一指标。显示数值作常规小数格式化，完整原始精度见"
        f"[D/VALIDATION_RESULTS.csv]({csvlink})。", "",
        "|profile / stream|GP尺度|H0尺度|pair增益均值|32项同时EB区间|选择|GP joint KL|H0 joint KL|所选 joint KL|",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['profile']} / {int(r['stream']):02}|"
            f"({float(r['GP_u']):g},{float(r['GP_v']):g})|{float(r['H0_scale']):g}|"
            f"{float(r['mean']):.9g}|[{float(r['lower']):.9g}, {float(r['upper']):.9g}]|"
            f"{r['selected_model']}|{r['GP_joint_KL']}|"
            f"{r['H0_joint_KL']}|{r['selected_joint_KL']}|")
    lines += [
        "", "区间公式、条件于训练和教师的独立性、完整81格支持范围，以及R=0的常数分支见[VALIDATION_THEORY.md](VALIDATION_THEORY.md)；原固定设计见[PLAN_D.md](PLAN_D.md)。统计保证条件于冻结的数值生成分布和候选损失，不把参考精度资格当作解析Gaussian模型的全误差证书。", "",
        f"本次图表来源和产物哈希见[SOURCE_HASHES.json]({baselink}/SOURCE_HASHES.json)，作用域仅为D补充图表及本页，与A/B/C figures分开。源码为script/structure_learning_validation_figures.py；只读检查参数为--check。旧D文件完全保持。", "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    normalize_text(DOC)


def build():
    assert not (BASE / "MANIFEST.json").exists(), "parent package already sealed; use a new version"
    rows, summary = source_guard()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    eb_plot(rows, summary)
    risk_plot(rows, summary)
    results_document(rows, summary)
    outputs = [OUT / name for name in OUTPUT_NAMES]+[DOC]
    sources = [D / name for name in SOURCE_NAMES]+[Path(__file__)]
    note = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="D supplementary figures and D_RESULTS only; not the final study manifest",
        sources={rel(p): sha(p) for p in sources},
        outputs={rel(p): sha(p) for p in outputs},
        source_rows=32, source_summary_rows=2, all_cases_shown=True,
        confidence="frozen 32 two-sided simultaneous 95% empirical Bernstein intervals",
        display_only_KL_zero_rule="abs(raw KL)<1e-12 drawn at zero, labelled approximately zero",
        raw_values_preserved_in_original_CSV_and_new_table=True,
        new_statistics_families=0, new_observations=0, new_fits=0,
        matplotlib=matplotlib.__version__, numpy=np.__version__)
    (OUT / "SOURCE_HASHES.json").write_text(json.dumps(note, ensure_ascii=False, indent=2)+"\n",
                                         encoding="utf-8", newline="\n")
    print(json.dumps(dict(figures=4, results_page=rel(DOC), all_EB_cases=32, new_science=0)))


def check():
    rows, summary = source_guard()
    note = read(OUT / "SOURCE_HASHES.json")
    for key in ("sources", "outputs"):
        for name, expected in note[key].items():
            assert sha(ROOT / name) == expected, name
    for path in [Path(__file__), DOC]+[OUT / name for name in OUTPUT_NAMES if name.endswith(".svg")]:
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n") and b"\r" not in raw, path
        assert all(line == line.rstrip() for line in raw.decode("utf-8").splitlines()), path
    assert note["source_rows"] == len(rows) == 32 and note["source_summary_rows"] == len(summary) == 2
    assert note["new_statistics_families"] == note["new_observations"] == note["new_fits"] == 0
    print(json.dumps(dict(check_pass=True, EB_cases=32, output_files=5, old_D_seals_preserved=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build() if args.build else check()
