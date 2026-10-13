"""Static B/E plots from frozen tables only; no new probability calculations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/structure_tradeoff_v2"
OUT = BASE / "figures"
SOURCES = [
    BASE / "B/TABLE.csv", BASE / "B/CONFIG.json", BASE / "B/INPUT.json",
    BASE / "B/FREEZE.json", BASE / "B/MANIFEST.json",
    BASE / "E/RESULTS.json", BASE / "E/CONFIG.json",
    BASE / "E/FREEZE.json", BASE / "E/RESULTS_FREEZE.json",
]
OUTPUTS = ["event_loss.png", "event_loss.svg", "block_budget.png",
           "block_budget.svg", "EB_TABLES.md"]
COLORS = {"null": "#64748b", "mild": "#167ca6", "contrast": "#c76a17"}
MARKERS = {"null": "o", "mild": "s", "contrast": "D"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_hash(path, expected):
    assert sha(path) == expected, f"source hash mismatch: {path}"


def source_guard():
    bm = load(BASE / "B/MANIFEST.json")
    for name, digest in bm["files"].items():
        assert_hash(ROOT / name, digest)
    ef = load(BASE / "E/RESULTS_FREEZE.json")
    for item in ef["files"]:
        assert_hash(ROOT / item["path"], item["sha256"])
    assert_hash(BASE / "E/FREEZE.json", ef["source_freeze_sha256"])
    with (BASE / "B/TABLE.csv").open(encoding="utf-8", newline="") as handle:
        b = list(csv.DictReader(handle))
    e = load(BASE / "E/RESULTS.json")["rows"]
    assert len(b) == 7 and len(e) == 12
    assert [int(row["b"]) for row in b] == [1, 2, 4, 8, 16, 32, 64]
    assert all(row["numerical_pass"] == "True" for row in b)
    assert all(int(row["d"]) == 128 and int(row["K"]) == 100 for row in b)
    assert all(row["n"] == 524288 for row in e)
    assert load(BASE / "B/CONFIG.json")["random_observations"] == 0
    return b, e


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.labelsize": 10.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#94a3b8",
        "xtick.color": "#334155", "ytick.color": "#334155",
        "text.color": "#172338", "axes.labelcolor": "#25364d",
        "grid.color": "#dbe3ed", "grid.linewidth": .65,
        "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "path",
        "svg.hashsalt": "structure_tradeoff_v2_EB",
    })


def logarithmic_dimensions(ax, values, label):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(min(values)*.82, max(values)*1.22)
    ax.set_xlabel(label)
    ax.grid(axis="both", alpha=.75)
    ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight", pad_inches=.18)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=.18,
                metadata={"Date": None, "Creator": "v2_tradeoff_figures.py"})
    plt.close(fig)


def event_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    fig.subplots_adjust(left=.075, right=.975, top=.72, bottom=.22, wspace=.29)
    fig.suptitle("Information loss hidden by total-count preservation",
                 x=.5, y=.975, fontsize=16, weight="semibold")
    fig.text(.5, .917,
             "Two fixed groups | NEG teacher, c = 0.9 | 524,288 independent observations per cell",
             ha="center", fontsize=10.5, color="#4b5b70")
    handles = []
    labels = []
    for profile in ("null", "mild", "contrast"):
        data = sorted((r for r in rows if r["profile"] == profile), key=lambda r: r["d"])
        d = np.array([r["d"] for r in data])
        shift = 100*np.array([r["p_hat"]-.5 for r in data])
        lo = 100*np.array([r["p_hat"]-r["cp95_bonferroni_lower"] for r in data])
        hi = 100*np.array([r["cp95_bonferroni_upper"]-r["p_hat"] for r in data])
        h = axes[0].errorbar(d, shift, yerr=[lo, hi], color=COLORS[profile],
                 marker=MARKERS[profile], markersize=6, capsize=4,
                 linewidth=1.65, elinewidth=1.2)
        axes[1].plot(d, [r["event_kl_simultaneous_lower"] for r in data],
                     color=COLORS[profile], marker=MARKERS[profile],
                     markersize=6, linewidth=1.65)
        handles.append(h)
        scales = data[0]["scales"]
        labels.append(f"{profile}: ({scales[0]:g}, {scales[1]:g})")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .884),
               ncol=3, columnspacing=2.8, handlelength=2.8)
    for ax in axes:
        logarithmic_dimensions(ax, [16, 64, 128, 512], "Dimension d (log scale)")
        ax.axvline(128, color="#b9c3d0", linestyle=":", linewidth=1, zorder=0)
    axes[0].axhline(0, color="#344256", linestyle="--", linewidth=1)
    axes[0].set_title("A  |  Sign-agreement event shift", loc="left", pad=13)
    axes[0].set_ylabel("P(same sign) - 1/2  (percentage points)")
    axes[0].set_ylim(-2.75, .58)
    axes[0].set_yticks([-2.5, -2, -1.5, -1, -.5, 0, .5])
    axes[0].text(.025, .04, "Error bars: 12 simultaneous 95% CP intervals",
                 transform=axes[0].transAxes, fontsize=9.1, color="#526174")
    axes[1].set_title("B  |  Event-information lower bound", loc="left", pad=13)
    axes[1].set_ylabel("Lower bound on full-vector KL  (nats)")
    axes[1].set_yscale("symlog", linthresh=1e-6, linscale=.7)
    axes[1].set_ylim(-1.6e-7, 1.8e-3)
    axes[1].set_yticks([0, 1e-6, 1e-5, 1e-4, 1e-3])
    axes[1].set_yticklabels(["0", r"$10^{-6}$", r"$10^{-5}$",
                            r"$10^{-4}$", r"$10^{-3}$"])
    axes[1].yaxis.set_minor_locator(mticker.NullLocator())
    axes[1].text(.035, .98, "Log-like y-axis; exact zeros retained",
                 transform=axes[1].transAxes, va="top", fontsize=9.1, color="#526174")
    fig.text(.075, .108,
             "Primary endpoints: d = 128, mild and contrast. KL lower bounds minimize kl(p || 1/2) over each CP interval.",
             fontsize=10, color="#334155")
    fig.text(.075, .06,
             "A small positive lower bound is not a full-KL estimate or evidence of a large task effect; zero is uninformative.",
             fontsize=10, color="#334155")
    save(fig, "event_loss")


def block_figure(rows):
    b = np.array([int(r["b"]) for r in rows])
    gamma = np.array([float(r["L_b_minus_1"]) for r in rows])
    inner = np.array([float(r["sum_block_Gaussian_TC_upper"]) for r in rows])
    blocktc = np.array([float(r["derived_block_TC_lower"]) for r in rows])
    kl = np.array([float(r["mixture_observed_KL_lower"]) for r in rows])
    tc = float(rows[0]["inherited_coordinate_TC_lower"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7))
    fig.subplots_adjust(left=.057, right=.98, top=.745, bottom=.24, wspace=.30)
    fig.suptitle("Within-block structure changes the particle-budget lower bound",
                 x=.5, y=.975, fontsize=16, weight="semibold")
    fig.text(.5, .92,
             "Homogeneous NEG teacher | d = 128, c = 0.9, sigma = 4 | one fixed shared equal-block partition",
             ha="center", fontsize=10.5, color="#4b5b70")
    for ax in axes:
        logarithmic_dimensions(ax, b, "Block size b (log scale)")
    axes[0].plot(b, gamma, color="#7253a3", marker="o", linewidth=1.8, markersize=6)
    axes[0].set_yscale("log")
    axes[0].set_ylim(.0047, 1.25)
    axes[0].set_yticks([.01, .1, 1])
    axes[0].set_yticklabels(["0.01", "0.1", "1"])
    axes[0].yaxis.set_minor_locator(mticker.NullLocator())
    axes[0].set_title("A  |  Mixture coefficient", loc="left", pad=13)
    axes[0].set_ylabel(r"$\gamma_b = L_b - 1$  (dimensionless)")
    axes[0].text(.045, .90, r"Allowance: $\gamma_b \log K$",
                 transform=axes[0].transAxes, fontsize=10, color="#7253a3")
    axes[1].plot(b, inner, color="#c76a17", marker="^", linewidth=1.8,
                 markersize=6, label="Within-block TC: upper bound")
    axes[1].plot(b, blocktc, color="#167ca6", marker="s", linewidth=1.8,
                 markersize=6, label="Between-block TC: lower bound")
    axes[1].axhline(tc, color="#64748b", linestyle="--", linewidth=1.2,
                   label="Inherited coordinate-TC lower bound")
    axes[1].set_title("B  |  Deducting within-block TC", loc="left", pad=13)
    axes[1].set_ylabel("Total correlation bounds  (nats)")
    axes[1].set_ylim(-.008, .205)
    axes[1].set_yticks([0, .05, .1, .15, .2])
    axes[1].legend(loc="upper left", fontsize=8.7, labelspacing=.35,
                   handlelength=1.8, borderaxespad=.2)
    axes[2].plot(b, kl, color="#1a857b", marker="D", linewidth=1.8,
                 markersize=6)
    axes[2].set_title("C  |  Observation-KL bound, K = 100", loc="left", pad=13)
    axes[2].set_ylabel("KL lower bound  (nats)")
    axes[2].set_ylim(-.0018, .031)
    axes[2].set_yticks([0, .01, .02, .03])
    axes[2].text(.16, .73, "b = 1: 0.02534949 nats",
                 transform=axes[2].transAxes, color="#176f68", fontsize=10)
    axes[2].text(.20, .36, "Zero = bound uninformative\n(not zero approximation error)",
                 transform=axes[2].transAxes, color="#334155", fontsize=10,
                 linespacing=1.6, ha="left")
    fig.text(.057, .115,
             "Frozen-table arithmetic only: inherited M2 coordinate-TC lower bound = 0.05798456 nats; no new probability data.",
             fontsize=10, color="#334155")
    fig.text(.057, .066,
             "These are necessary constraints, not sufficient budgets, runtime guarantees, or empirical approximation errors.",
             fontsize=10, color="#334155")
    save(fig, "block_budget")


def readable_table(b, e):
    lines = [
        "# B/E 可读合并表：结构预算与错对称化信息损失", "",
        "仅从已封存 B/TABLE.csv 与 E/RESULTS.json 转录，不新增积分、概率样本、拟合、epsilon门或置信区间。两包对象不同：B继承同尺度 NEG/sigma4 的旧M2下界，E使用三个预定两组尺度教师；不能将B的旧数值下界移植为异质教师认证。所有KL和TC单位为完整向量nats。", "",
        "## E：符号事件揭示总计数看不到的组分配信息", "",
        "每格524,288独立观察；T/V零值分别使用独立公平coin。完整对称化在同一随机化通道下的事件基线有限d精确为1/2。CP为12项Bonferroni同时95%；KL下界是该区间上kl(p||1/2)的最小值，是完整错配KL的粗事件下界，不是完整KL点估计。", "",
        "|profile (sA,sB)|d|成功数|事件概率|同时CP区间|事件KL下界|预定主终点|",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for r in e:
        s = r["scales"]
        primary = ("通过" if r["primary_pass"] else "未通过") if r["primary"] else "描述/对照"
        lines.append(f"|{r['profile']} ({s[0]:g},{s[1]:g})|{r['d']}|{r['successes']}|"
            f"{r['p_hat']:.9f}|[{r['cp95_bonferroni_lower']:.9f}, {r['cp95_bonferroni_upper']:.9f}]|"
            f"{r['event_kl_simultaneous_lower']:.10g}|{primary}|")
    lines += [
        "", "两主终点是d128的mild与contrast，规则仅为下界严格大于0；没有实质大小阈值。小正下界不等于大实际影响，不能推出自然任务收益。零下界表示该事件区间没有给出正KL证据，并不是完整错配KL为零的统计证明。null教师本身完整交换是解析事实；有限样本区间只作保留对照。", "",
        "图：event_loss.png/svg。横轴按log尺度列出真实d；左图为事件概率减1/2的百分点偏移与CP误差条；右图为KL下界，symlog保留真实零值，不把零替换为任意小正数。连线仅作视觉引导。", "",
        "## B：固定分区内部结构与K=100下界", "",
        "固定d128、c=.9、K100、同尺度sigma4教师。所有分量共享同一非重叠等大分区。gamma=L_b−1；块内TC扣除项取Gaussian数据处理上界，不能当作实际块内信息。块间TC下界为max(0,继承的全TC下界−该扣除上界)；观察KL下界再扣gamma log100并截至0。数值和截断均来自已冻CSV，未新增阈值。", "",
        "|块大小b|块数|gamma|块内TC总和上界|块间TC下界|gamma log100|K100观察KL下界|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in b:
        vals = [r["b"], r["blocks"]] + [f"{float(r[k]):.10g}" for k in
            ["L_b_minus_1", "sum_block_Gaussian_TC_upper", "derived_block_TC_lower",
             "mixture_allowance_gamma_logK", "mixture_observed_KL_lower"]]
        lines.append("|"+"|".join(vals)+"|")
    lines += [
        "", "继承的坐标TC下界为0.057984557899112235 nats，来自旧M2原24-event CP family；这里没有新置信区间。b=1时K100的KL下界为0.0253494935889604；b≥2时这个代入下界为0，只能说界无信息，不能说真实近似误差为0或少量粒子已充分。b≥32时继承的块间TC代入下界也已为0，同样保留。", "",
        "图：block_budget.png/svg。三个面板分别显示gamma、块内TC扣除上界与剩余块间TC下界、K100观察KL下界。它们是必要预算的确定性算术，不能解释成训练曲线、运行成本或方法实际误差。", "",
        "## 来源与核验", "",
        "SOURCE_HASHES.json记录本脚本、B/E源及本轮五个图表文件哈希，仅覆盖B/E图表，不是整个v2最终manifest。生成前核验原B/MANIFEST与E/RESULTS_FREEZE；原冻结完全保持。", "",
        "只读核验：.venv/Scripts/python -X utf8 script/v2_tradeoff_figures.py --check。图表整理没有新科学样本、积分节点或拟合。", "",
    ]
    (OUT / "EB_TABLES.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build():
    b, e = source_guard()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    event_figure(e)
    block_figure(b)
    readable_table(b, e)
    receipt = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="B/E figures and readable tables only; not final v2 manifest",
        source_files=[dict(path=rel(p), sha256=sha(p)) for p in SOURCES + [Path(__file__)]],
        outputs=[dict(path=rel(OUT / name), sha256=sha(OUT / name)) for name in OUTPUTS],
        matplotlib=matplotlib.__version__, numpy=np.__version__, python=platform.python_version(),
        new_probability_observations=0, new_integration_nodes=0, new_fits=0,
        event_CI="original 12-family simultaneous 95% Clopper-Pearson",
        zero_handling="exact zeros retained; E KL symlog linear neighborhood 1e-6",
        B_CI="inherited original M2 24-event family; no new interval",
    )
    (OUT / "SOURCE_HASHES.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(created=OUTPUTS, source_count=len(receipt["source_files"]),
                          science_samples=0), ensure_ascii=False))


def check():
    b, e = source_guard()
    receipt = load(OUT / "SOURCE_HASHES.json")
    for item in receipt["source_files"] + receipt["outputs"]:
        assert_hash(ROOT / item["path"], item["sha256"])
    assert len(receipt["outputs"]) == 5
    assert receipt["new_probability_observations"] == receipt["new_integration_nodes"] == receipt["new_fits"] == 0
    assert len([x for x in e if x["primary"]]) == 2
    assert all(float(row["mixture_observed_KL_lower"]) == 0 for row in b[1:])
    print(json.dumps(dict(check_pass=True, event_rows=len(e), block_rows=len(b),
                          generated_files=5, science_samples=0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = p.parse_args()
    build() if args.build else check()
