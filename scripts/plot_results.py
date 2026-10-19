"""Report figures from saved results only (no GPU, no models). Figures whose inputs are missing are skipped.

    python -m scripts.plot_results [--results results] [--out report/figures]

Every figure is written as PDF (for the report) and PNG. The CSV/JSONL files they are drawn from
remain the exact-value reference for every plotted number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import read_jsonl, repo_path  # noqa: E402

# Categorical slots in fixed order (validated default palette); status colours only for outcomes.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OUTCOME = {"better": "#0ca30c", "tie": "#c3c2b7", "wrong": "#d03b3b"}
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SAFETY_LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "OVER_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS"]
POLICIES = ["sft", "dpo", "ppo", "grpo"]


def style() -> None:
    """Shared style: light surface, hairline solid grid, small type sized for an 8-page paper."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titlesize": 8.5, "axes.titlecolor": INK, "axes.labelcolor": SECONDARY, "font.size": 7.5,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False, "legend.fontsize": 7,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, out: Path, name: str) -> None:
    """Write PDF + PNG and close the figure."""
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def load_jsonl(path: Path) -> pd.DataFrame | None:
    """JSONL file as a DataFrame, or None if it does not exist yet."""
    return pd.DataFrame(read_jsonl(path)) if path.exists() else None


def load_csv(path: Path) -> pd.DataFrame | None:
    """CSV file as a DataFrame, or None if it does not exist yet."""
    return pd.read_csv(path) if path.exists() else None


def panels(n: int, cols: int = 4, height: float = 1.7):
    """Grid of small panels with unused cells hidden; returns (figure, flat array of axes)."""
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.75 * cols, height * rows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def legend_above(fig, ax, ncol: int) -> None:
    """Shared legend in its own band above the panels (call after tight_layout)."""
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 1.0))


def trajectories(df: pd.DataFrame, metrics: dict, group: str | None = None):
    """Small multiples, one metric per panel (never two y-scales on one axis); one line per `group`."""
    fig, axes = panels(len(metrics))
    groups = [(None, df)] if group is None else list(df.groupby(group, sort=False))
    for ax, (col, title) in zip(axes, metrics.items()):
        for i, (label, d) in enumerate(groups):
            ax.plot(d["update"], d[col], color=SERIES[i], label=label, marker="o" if len(d) <= 10 else None)
        ax.set_title(title)
        ax.set_xlabel("update")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    if group is not None:
        legend_above(fig, axes[0], len(groups))
    return fig


def stacked_bars(ax, table: pd.DataFrame, colors: dict, horizontal: bool = True) -> None:
    """Stacked shares with a thin surface gap between segments."""
    left = np.zeros(len(table))
    for col, color in colors.items():
        values = table[col].to_numpy() if col in table else np.zeros(len(table))
        kwargs = dict(color=color, label=col, edgecolor=SURFACE, linewidth=1.0)
        if horizontal:
            ax.barh(table.index.astype(str), values, left=left, height=0.7, **kwargs)
        else:
            ax.bar(table.index.astype(str), values, bottom=left, width=0.7, **kwargs)
        left += values


# ----------------------------------------------------------------------------- figures

def fig_ppo(results: Path, out: Path) -> None:
    """PPO standard trajectories, beta_KL / epsilon fork trajectories and the cached clipping geometry."""
    log = load_jsonl(results / "task2_ppo/train/standard/update_log.jsonl")
    if log is not None:
        save(trajectories(log, {
            "reward": "learned reward", "kl": "KL to reference", "policy_loss": "policy loss", "value_loss": "value loss",
            "entropy": "entropy", "clip_fraction": "clip fraction", "grad_norm": "policy grad norm",
            "response_tokens": "response length", "explained_variance": "critic explained var.",
        }), out, "task2_ppo_standard_trajectories")
    kl = load_csv(results / "task2_ppo/kl_trajectories.csv")
    if kl is not None:
        save(trajectories(kl, {"reward": "learned reward", "kl": "KL to reference", "entropy": "entropy",
                               "response_tokens": "response length"}, group="fork"), out, "task2_ppo_kl_forks")
    clip = load_csv(results / "task2_ppo/clipping_trajectories.csv")
    if clip is not None:
        save(trajectories(clip, {"clip_fraction": "clip fraction", "step_kl_old_new": "per-update KL(old||new)",
                                 "grad_norm_max": "max grad norm", "reward": "learned reward"}, group="fork"), out, "task2_ppo_clip_forks")
    cached = load_csv(results / "task2_ppo/clipping_cached.csv")
    if cached is not None:
        fig, axes = panels(2, cols=2, height=1.9)
        for i, (eps, d) in enumerate(cached.groupby("epsilon")):
            axes[0].plot(d["step"], d["clip_fraction"], color=SERIES[i], marker="o", label=f"eps={eps:g}")
            axes[1].plot(d["step"], d["affected_fraction"], color=SERIES[i], marker="o", label=f"eps={eps:g}")
        for ax, title in zip(axes, ["clip fraction (cached batch)", "affected-token fraction"]):
            ax.set_title(title)
            ax.set_xlabel("optimizer step on cached batch")
        axes[0].legend()
        fig.tight_layout()
        save(fig, out, "task2_ppo_clip_cached")


def fig_grpo(results: Path, out: Path) -> None:
    """GRPO standard trajectories, group-size study and normalization gradient allocation."""
    log = load_jsonl(results / "task3_grpo/train/standard/update_log.jsonl")
    if log is not None:
        save(trajectories(log, {
            "reward": "learned reward", "kl": "KL to reference", "reward_std_within_group": "within-group reward std",
            "frac_uninformative_groups": "uninformative groups", "policy_loss": "policy loss", "grad_norm": "grad norm",
            "entropy": "entropy", "response_tokens": "response length",
        }), out, "task3_grpo_standard_trajectories")
    summary = load_csv(results / "task3_grpo/group_size/group_size_summary.csv")
    if summary is not None:
        partition = "random_mean" if (summary["partition"] == "random_mean").any() else "contiguous"
        s = summary[summary["partition"] == partition]
        metrics = {"informative_rate_practical": "informative groups (practical)", "mean_group_std": "mean within-group std",
                   "advantage_var": "advantage variance", "sign_agreement_k8": "sign agreement with K=8"}
        fig, axes = panels(len(metrics))
        for ax, (col, title) in zip(axes, metrics.items()):
            for i, (subset, d) in enumerate(s.groupby("subset", sort=False)):
                ax.plot(d["K"], d[col], color=SERIES[i], marker="o", label=subset)
            ax.set_xticks(sorted(s["K"].unique()))
            ax.set_title(title)
            ax.set_xlabel("group size K")
        fig.tight_layout()
        legend_above(fig, axes[0], ncol=4)
        save(fig, out, "task3_group_size")
    frames = [load_jsonl(results / f"task3_grpo/train/{name}/completions.jsonl") for name in ("norm_grpo", "norm_dr_grpo")]
    if all(f is not None for f in frames):
        fig, axes = panels(2, cols=2, height=2.0)
        for i, (label, d) in enumerate(zip(["GRPO (1/T)", "Dr. GRPO (1/L_max)"], frames)):
            d = d[d["in_loss"] & (d["advantage"].abs() > 1e-8)]
            axes[0].scatter(d["response_tokens"], d["grad_per_token"], s=10, color=SERIES[i], edgecolor=SURFACE, linewidth=0.5, label=label)
            axes[1].scatter(d["response_tokens"], d["grad_mass"], s=10, color=SERIES[i], edgecolor=SURFACE, linewidth=0.5, label=label)
        for ax, title in zip(axes, ["gradient weight per token", "gradient mass per completion"]):
            ax.set_title(title)
            ax.set_xlabel("completion length (tokens)")
        axes[0].legend()
        fig.tight_layout()
        save(fig, out, "task3_normalization_gradient_allocation")


def fig_dpo(results: Path, out: Path) -> None:
    """DPO beta summary bars and per-stratum preference accuracy."""
    summary = load_csv(results / "task1_dpo/beta_summary.csv")
    if summary is not None:
        metrics = {"preference_accuracy": "held-out pref. accuracy", "kl_token_mean": "KL to reference",
                   "reward_mean": "reward-model score", "length_mean": "response length"}
        fig, axes = panels(len(metrics), height=2.0)
        for ax, (col, title) in zip(axes, metrics.items()):
            ax.bar(summary["condition"], summary[col], color=SERIES[0], width=0.6)
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        fig.tight_layout()
        save(fig, out, "task1_dpo_beta_summary")
    strata = load_csv(results / "task1_dpo/length_strata.csv")
    if strata is not None:
        table = strata.pivot(index="stratum", columns="model", values="preference_accuracy")
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        width = 0.38
        for i, model in enumerate(table.columns):
            ax.bar(np.arange(len(table)) + (i - 0.5) * width, table[model], width=width * 0.95, color=SERIES[i], label=model)
        ax.set_xticks(np.arange(len(table)), table.index)
        ax.set_title("held-out preference accuracy by length stratum")
        ax.legend()
        fig.tight_layout()
        save(fig, out, "task1_length_strata")


def fig_safety(results: Path, out: Path) -> None:
    """Judge-label shares per policy on SAFE and UNSAFE prompts."""
    counts = load_csv(results / "task4_safety/category_counts.csv")
    if counts is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.9), sharey=True)
    for ax, cls in zip(axes, ["SAFE", "UNSAFE"]):
        table = counts[counts["benchmark_class"] == cls].groupby("policy")[SAFETY_LABELS].sum()
        table = table.reindex([p for p in POLICIES if p in table.index])
        stacked_bars(ax, table.div(table.sum(axis=1), axis=0), dict(zip(SAFETY_LABELS, SERIES)))
        ax.set_title(f"{cls} prompts: share of judge labels")
        ax.set_xlim(0, 1)
    axes[0].invert_yaxis()  # policies top-to-bottom in training order
    fig.tight_layout()
    legend_above(fig, axes[0], ncol=5)
    save(fig, out, "task4_safety_labels")


def fig_feedback(results: Path, out: Path) -> None:
    """better / tie / wrong shares per controlled pair for the verifier and the judge."""
    summary = load_csv(results / "task5_feedback/diagnostics/pair_summary.csv")
    if summary is None:
        return
    summary = summary[summary["mechanism"].isin(["rlvr", "rlaif"])]
    table = summary.set_index(summary["pair"] + " / " + summary["mechanism"])
    table = table.rename(columns={f"{k}_rate": k for k in OUTCOME})
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    stacked_bars(ax, table, OUTCOME)
    ax.set_xlim(0, 1)
    ax.set_title("controlled pairs: share preferring the better response")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, out, "task5_diagnostic_preferences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    results, out = repo_path(args.results), repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    for make in (fig_dpo, fig_ppo, fig_grpo, fig_safety, fig_feedback):
        make(results, out)


if __name__ == "__main__":
    main()
