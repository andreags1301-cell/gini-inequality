"""Residual diagnostics that motivate the fixed-effects + clustered-SE spec.

  * Breusch-Pagan  - heteroskedasticity of the pooled OLS residuals.
  * Breusch-Godfrey - serial autocorrelation of the within (FE) residuals.

The autocorrelation test is the key one: Gini series are highly persistent, so
errors are correlated over time within a country, which is exactly what the
country-clustered standard errors correct for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

from . import config


def breusch_pagan(df: pd.DataFrame, y: str, x: str) -> dict:
    """Breusch-Pagan (Koenker studentized) test on pooled OLS residuals.

    H0: constant residual variance. Matches R's ``lmtest::bptest`` default.
    """
    model = sm.OLS(df[y], sm.add_constant(df[x])).fit()
    lm, lm_p, f, f_p = het_breuschpagan(model.resid, model.model.exog)
    return {"stat": lm, "pvalue": lm_p, "df": 1}


def breusch_godfrey_fe(df: pd.DataFrame, y: str, x: str, order: int = 1) -> dict:
    """Breusch-Godfrey LM test for serial correlation on within residuals.

    Country-demeans y and x (the within transformation), regresses the residual
    on the demeaned regressor plus ``order`` within-country lags of the residual,
    and forms the LM statistic ``n * R^2 ~ chi^2(order)``.

    H0: no serial correlation. This is the Python analogue of R's ``plm::pbgtest``
    (which defaults to a higher lag order); both reject H0 overwhelmingly here.
    """
    d = df[["cnt", "year", y, x]].sort_values(["cnt", "year"]).copy()

    def demean(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g[y] = g[y] - g[y].mean()
        g[x] = g[x] - g[x].mean()
        return g

    d = d.groupby("cnt", group_keys=False)[["cnt", "year", y, x]].apply(demean)

    # Within residuals from the demeaned bivariate regression (no intercept).
    beta = (d[x] * d[y]).sum() / (d[x] ** 2).sum()
    d["resid"] = d[y] - beta * d[x]

    # Auxiliary regression: resid ~ x + lag_1..lag_order(resid), within country.
    aux = d[["cnt", x, "resid"]].copy()
    for lag in range(1, order + 1):
        aux[f"lag{lag}"] = d.groupby("cnt")["resid"].shift(lag)
    aux = aux.dropna()

    exog = sm.add_constant(aux[[x] + [f"lag{lag}" for lag in range(1, order + 1)]])
    r2 = sm.OLS(aux["resid"], exog).fit().rsquared
    n = len(aux)
    lm = n * r2
    p = stats.chi2.sf(lm, df=order)
    return {"stat": lm, "pvalue": p, "df": order}


def run_all(frames: dict, order: int = 1) -> dict:
    """Run both diagnostics for every model."""
    out = {"breusch_pagan": {}, "breusch_godfrey": {}}
    for key, spec in config.MODELS.items():
        y, x = spec["y"], spec["x"]
        df = frames["AB"] if key in ("A", "B") else frames["C"]
        out["breusch_pagan"][key] = breusch_pagan(df, y, x)
        out["breusch_godfrey"][key] = breusch_godfrey_fe(df, y, x, order=order)
    return out
