"""Generate publication figures for the CEPS tutoring analysis."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
FIGURE_DIR = PROJECT_ROOT / "figures"

sys.path.insert(0, str(SCRIPT_DIR))

from models import clustered_regression, covariates_for_model  # noqa: E402


METHOD_ORDER = ["DID", "IPW-DID", "DR-DID"]
METHOD_STYLE = {
    "DID": {"color": "#2F5597", "marker": "o"},
    "IPW-DID": {"color": "#C00000", "marker": "s"},
    "DR-DID": {"color": "#4D4D4D", "marker": "^"},
}

BALANCE_LABELS = {
    "z_total_bl": "基线总成绩",
    "z_cog_bl": "基线认知能力",
    "male": "男生",
    "age": "年龄",
    "han": "汉族",
    "only_child": "独生子女",
    "father_edu": "父亲教育",
    "mother_edu": "母亲教育",
    "family_econ": "家庭经济状况",
    "parent_income_relative": "家庭相对收入",
    "parent_child_time": "家长时间投入",
}

MAIN_LABELS = {
    "math_tutoring_to_math_score": "数学补习 -> 数学成绩",
    "english_tutoring_to_english_score": "英语补习 -> 英语成绩",
    "chinese_tutoring_to_chinese_score": "语文补习 -> 语文成绩",
    "any_extracurricular_to_total_score": "任意课外补习 -> 总成绩",
    "core_tutoring_to_total_score": "任意主课补习 -> 总成绩",
    "core_tutoring_to_cognition": "任意主课补习 -> 一般认知能力",
    "core_tutoring_to_negative_emotion": "任意主课补习 -> 平均负面情绪",
    "core_tutoring_to_high_negative_emotion": "任意主课补习 -> 高负面情绪",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def save_figure(fig, name):
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


def p_label(pvalue):
    if pvalue < 0.001:
        return "p<0.001"
    return f"p={pvalue:.3f}"


def plot_balance_single(data, column, title, color, filename, xlim):
    ordered = data.assign(abs_smd=data[column].abs()).sort_values("abs_smd")
    labels = [BALANCE_LABELS.get(c, c) for c in ordered["covariate"]]
    values = ordered[column].to_numpy()
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y, values, color=color, alpha=0.86)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.axvline(0.1, color="#777777", linewidth=1, linestyle=":")
    ax.axvline(-0.1, color="#777777", linewidth=1, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("标准化均值差")
    ax.set_title(title)
    ax.set_xlim(xlim)
    save_figure(fig, filename)


def plot_balance():
    data = pd.read_csv(PROJECT_ROOT / "outputs" / "balance_diagnostics.csv")
    max_abs = max(data["raw_smd"].abs().max(), data["ipw_smd"].abs().max(), 0.1)
    xlim = (-max_abs * 1.2, max_abs * 1.2)
    plot_balance_single(
        data,
        "raw_smd",
        "IPW 前协变量平衡",
        "#C00000",
        "figure1_balance_before.png",
        xlim,
    )
    plot_balance_single(
        data,
        "ipw_smd",
        "IPW 后协变量平衡",
        "#2F5597",
        "figure1_balance_after.png",
        xlim,
    )


def plot_coefs(data, analyses, title, filename, xlim=None):
    rows = []
    for i, analysis in enumerate(analyses):
        for method in METHOD_ORDER:
            row = data[(data["analysis"] == analysis) & (data["method"] == method)]
            if row.empty:
                continue
            item = row.iloc[0].to_dict()
            item["label"] = MAIN_LABELS.get(analysis, analysis)
            item["base_y"] = i
            rows.append(item)
    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    offsets = {"DID": -0.18, "IPW-DID": 0.0, "DR-DID": 0.18}
    for method in METHOD_ORDER:
        sub = plot_data[plot_data["method"] == method]
        if sub.empty:
            continue
        y = sub["base_y"].to_numpy() + offsets[method]
        style = METHOD_STYLE[method]
        xerr = np.vstack(
            [
                sub["coef"].to_numpy() - sub["ci_lower"].to_numpy(),
                sub["ci_upper"].to_numpy() - sub["coef"].to_numpy(),
            ]
        )
        ax.errorbar(
            sub["coef"],
            y,
            xerr=xerr,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.4,
            capsize=3,
            markersize=5,
            label=method,
        )
        for _, r in sub.iterrows():
            ax.text(
                r["ci_upper"] + 0.006,
                r["base_y"] + offsets[method],
                f"{r['coef']:.3f}, {p_label(r['pvalue'])}",
                va="center",
                fontsize=9,
                color=style["color"],
            )

    labels = [MAIN_LABELS.get(a, a) for a in analyses]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.45, -0.45)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("估计系数与 95% 置信区间")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
    )
    save_figure(fig, filename)


def plot_main_results():
    main = pd.read_csv(PROJECT_ROOT / "outputs" / "main_results.csv")
    academic = [
        "math_tutoring_to_math_score",
        "english_tutoring_to_english_score",
        "chinese_tutoring_to_chinese_score",
        "any_extracurricular_to_total_score",
        "core_tutoring_to_total_score",
    ]
    nonacademic = [
        "core_tutoring_to_cognition",
        "core_tutoring_to_negative_emotion",
        "core_tutoring_to_high_negative_emotion",
    ]
    plot_coefs(
        main,
        academic,
        "实证结果：学业成绩",
        "figure2_main_academic.png",
        xlim=(-0.16, 0.25),
    )
    plot_coefs(
        main,
        nonacademic,
        "实证结果：认知水平和情绪状态",
        "figure2_main_cognition_emotion.png",
        xlim=(-0.10, 0.16),
    )


def math_cross_subject_results(control_other_subjects):
    fall = pd.read_parquet(PROJECT_ROOT / "dataset" / "processed" / "tutoring_fall2013_sample.parquet")
    data = fall[(fall["new_core"] == 1) | (fall["never_core"] == 1)].copy()
    specs = [
        ("delta_math", "z_math_bl", "数学成绩"),
        ("delta_chinese", "z_chinese_bl", "语文成绩"),
        ("delta_english", "z_english_bl", "英语成绩"),
    ]
    rows = []
    controls = ["new_chinese", "new_english"] if control_other_subjects else []
    for outcome, baseline, label in specs:
        covariates = covariates_for_model([baseline], controls)
        result = clustered_regression(
            data,
            outcome,
            "new_math",
            covariates,
            method="DID",
        )
        rows.append(
            {
                "label": label,
                "method": "DID",
                "coef": result["coef"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "pvalue": result["pvalue"],
            }
        )
    return pd.DataFrame(rows)


def plot_math_cross_subject(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(data))
    xerr = np.vstack(
        [
            data["coef"].to_numpy() - data["ci_lower"].to_numpy(),
            data["ci_upper"].to_numpy() - data["coef"].to_numpy(),
        ]
    )
    ax.errorbar(
        data["coef"],
        y,
        xerr=xerr,
        fmt="o",
        color="#2F5597",
        ecolor="#2F5597",
        capsize=4,
        elinewidth=1.5,
        markersize=6,
    )
    for _, r in data.iterrows():
        ax.text(
            r["ci_upper"] + 0.006,
            r.name,
            f"{r['coef']:.3f}, {p_label(r['pvalue'])}",
            va="center",
            fontsize=10,
            color="#2F5597",
        )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_ylim(len(data) - 0.45, -0.45)
    ax.set_xlim(-0.08, 0.18)
    ax.set_xlabel("数学补习估计系数与 95% 置信区间")
    ax.set_title(title)
    save_figure(fig, filename)


def plot_cross_subject_comparison():
    no_controls = math_cross_subject_results(control_other_subjects=False)
    with_controls = math_cross_subject_results(control_other_subjects=True)
    plot_math_cross_subject(
        no_controls,
        "数学补习效应：未控制其他主课补习",
        "figure3_math_without_subject_controls.png",
    )
    plot_math_cross_subject(
        with_controls,
        "数学补习效应：控制其他主课补习",
        "figure3_math_with_subject_controls.png",
    )


def main():
    setup_style()
    plot_balance()
    plot_main_results()
    plot_cross_subject_comparison()


if __name__ == "__main__":
    main()
