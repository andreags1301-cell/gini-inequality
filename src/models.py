"""Estimators: pooled OLS, between, within fixed effects, and the reaction model.

Each of the three models (A, B, C) is estimated with three estimators that
exploit a different source of variation:

  * Pooled OLS  - all 523 country-year observations jointly (HC3 robust SE).
  * Between     - regression on the 32 country time-averages.
  * Within FE   - country-demeaned data, country-clustered SE (main spec).

This is the Python counterpart of the ``lm`` / ``plm`` calls in the legacy R
script, using statsmodels for OLS and linearmodels for the panel estimators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

from . import config


@dataclass
class Fit:
    """Minimal, estimator-agnostic container for a single coefficient result."""

    label: str
    beta: float
    se: float
    r2: float
    n: int
    tstat: float
    pvalue: float


def pooled_ols(df: pd.DataFrame, y: str, x: str, label: str) -> Fit:
    """Pooled OLS with HC3 heteroskedasticity-robust standard errors."""
    model = sm.OLS(df[y], sm.add_constant(df[x])).fit(cov_type="HC3")
    return Fit(
        label=label,
        beta=model.params[x],
        se=model.bse[x],
        r2=model.rsquared,
        n=int(model.nobs),
        tstat=model.tvalues[x],
        pvalue=model.pvalues[x],
    )


def between_ols(df_between: pd.DataFrame, y: str, x: str, label: str) -> Fit:
    """Between estimator: OLS on the country time-averages (classical SE)."""
    model = sm.OLS(df_between[y], sm.add_constant(df_between[x])).fit()
    return Fit(
        label=label,
        beta=model.params[x],
        se=model.bse[x],
        r2=model.rsquared,
        n=int(model.nobs),
        tstat=model.tvalues[x],
        pvalue=model.pvalues[x],
    )


def within_fe(df: pd.DataFrame, y: str, x: str, label: str) -> tuple[Fit, object]:
    """Within (fixed-effects) estimator with country-clustered standard errors.

    Returns both the :class:`Fit` summary and the underlying linearmodels result
    (the latter is reused for diagnostics and plotting).
    """
    panel = df.set_index(["cnt", "year"])
    res = PanelOLS(panel[y], panel[[x]], entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    fit = Fit(
        label=label,
        beta=res.params[x],
        se=res.std_errors[x],
        r2=res.rsquared,  # within R-squared
        n=int(res.nobs),
        tstat=res.tstats[x],
        pvalue=res.pvalues[x],
    )
    return fit, res


def rank_preservation_test(fit: Fit) -> dict:
    """Test H0: beta = 1 (redistribution merely rescales, preserving ranks).

    Uses the clustered SE of the within estimate, matching the R script.
    """
    t = (fit.beta - 1.0) / fit.se
    df_resid = fit.n - 2
    p = 2 * stats.t.sf(abs(t), df=df_resid)
    return {"label": fit.label, "beta": fit.beta, "se": fit.se, "t": t, "p": p}


def reaction_model(df_between: pd.DataFrame) -> dict:
    """Redistribution reaction model (between regression on 32 countries).

    ``reduc_public = ehearng - ehdinc_`` is the number of Gini points the fiscal
    system removes. Regressing it on market inequality tests whether countries
    redistribute *more* where inequality is greater.
    """
    model = sm.OLS(
        df_between["reduc_public"], sm.add_constant(df_between["ehearng"])
    ).fit()
    return {
        "beta": model.params["ehearng"],
        "se": model.bse["ehearng"],
        "r2": model.rsquared,
        "pvalue": model.pvalues["ehearng"],
        "n": int(model.nobs),
    }


def run_all(frames: dict) -> dict:
    """Estimate every model with every estimator and return a tidy structure."""
    results: dict = {"pooled": {}, "between": {}, "within": {}, "within_res": {}}
    for key, spec in config.MODELS.items():
        y, x, label = spec["y"], spec["x"], spec["label"]
        df = frames["AB"] if key in ("A", "B") else frames["C"]

        results["pooled"][key] = pooled_ols(df, y, x, label)
        results["between"][key] = between_ols(frames["between"], y, x, label)
        fit, res = within_fe(df, y, x, label)
        results["within"][key] = fit
        results["within_res"][key] = res

    results["rank"] = {
        k: rank_preservation_test(results["within"][k]) for k in ("B", "C")
    }
    results["reaction"] = reaction_model(frames["between"])
    return results
