"""Figures: the pooled-OLS baseline scatter and the fixed-effects scatter.

Reproduces the two ggplot graphs from the legacy R script using matplotlib.
Both figures focus on Model B (the main specification): Gini of disposable
income (after redistribution) against Gini of market income (before).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import config

_X, _Y = "ehearng", "ehdinc_"
_XLAB = "Gini market income (ehearng) - before redistribution"
_YLAB = "Gini disposable income (ehdinc_) - after redistribution"


def _assign_period(year: int) -> str:
    for name, upper, _ in config.PERIODS:
        if year <= upper:
            return name
    return config.PERIODS[-1][0]


def pooled_scatter(df_ab: pd.DataFrame, pooled_fit, ax=None):
    """Figure 1 - pooled OLS baseline (Step 1)."""
    ax = ax or plt.subplots(figsize=(9, 6))[1]
    x, y = df_ab[_X], df_ab[_Y]

    ax.scatter(x, y, s=28, color="#2166ac", alpha=0.6, edgecolor="none")

    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, pooled_fit.beta * xs + (y.mean() - pooled_fit.beta * x.mean()),
            color="blue", linewidth=1.6, label="OLS fit")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "--", color="red", linewidth=1.4,
            label=r"$\beta=1$ benchmark (no redistribution)")

    ax.set_title(
        "Step 1 - Pooled OLS: Gini after vs before redistribution\n"
        rf"$\it{{y = \alpha + \beta X}}$   |   $\beta$ = {pooled_fit.beta:.3f} "
        rf"(HC3 SE = {pooled_fit.se:.3f})   |   N = {pooled_fit.n}",
        fontsize=12, fontweight="bold")
    ax.set_xlabel(_XLAB)
    ax.set_ylabel(_YLAB)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    return ax


def fe_scatter(df_ab: pd.DataFrame, within_fit, ax=None):
    """Figure 2 - fixed-effects fit, points coloured by period (Step 3)."""
    ax = ax or plt.subplots(figsize=(9, 6))[1]
    d = df_ab.dropna(subset=[_X, _Y, "year"]).copy()
    d["period"] = d["year"].apply(_assign_period)

    for name, _, colour in config.PERIODS:
        sub = d[d["period"] == name]
        ax.scatter(sub[_X], sub[_Y], s=34, alpha=0.85, color=colour, label=name)

    lim = [min(d[_X].min(), d[_Y].min()), max(d[_X].max(), d[_Y].max())]
    ax.plot(lim, lim, "--", color="red", linewidth=1.4,
            label=r"$\beta=1$ benchmark")

    intercept = d[_Y].mean() - within_fit.beta * d[_X].mean()
    xs = np.linspace(d[_X].min(), d[_X].max(), 100)
    ax.plot(xs, within_fit.beta * xs + intercept, color="black", linewidth=1.7,
            label="FE fit")

    ax.set_title(
        "Step 3 - Fixed Effects: Gini after vs before redistribution\n"
        rf"$\it{{y_{{it}} = \alpha_i + \beta X_{{it}} + u_{{it}}}}$   |   "
        rf"$\beta$ = {within_fit.beta:.3f} (clustered SE = {within_fit.se:.3f})"
        rf"   |   N = {within_fit.n}",
        fontsize=12, fontweight="bold")
    ax.set_xlabel(_XLAB)
    ax.set_ylabel(_YLAB)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def save_all(frames: dict, results: dict, figures_dir=config.FIGURES_DIR,
             assets_dir=config.ROOT / "assets") -> list:
    """Render both figures to results/figures (and copy Fig. 2 to assets/)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(9, 6))
    pooled_scatter(frames["AB"], results["pooled"]["B"], ax=ax)
    fig.tight_layout()
    p1 = figures_dir / "fig1_pooled_ols.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    fig, ax = plt.subplots(figsize=(9, 6))
    fe_scatter(frames["AB"], results["within"]["B"], ax=ax)
    fig.tight_layout()
    p2 = figures_dir / "fig2_fixed_effects.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    fig.savefig(assets_dir / "fig2_fixed_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    return saved
