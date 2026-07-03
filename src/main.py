"""End-to-end analysis pipeline.

Run from the repository root with the project environment active:

    python -m src.main

It reproduces the full analysis of the legacy R script:
  1. Load and reshape the EU-SILC Gini data.
  2. Estimate every model (A, B, C) with pooled OLS, between and within FE.
  3. Run residual diagnostics (Breusch-Pagan, Breusch-Godfrey).
  4. Save summary tables and intermediate data to ``results/``.
  5. Render the two figures to ``results/figures/``.
"""
from __future__ import annotations

import pandas as pd

from . import config, data_prep, diagnostics, models, plots

_RULE = "=" * 70


def _print_header(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def _summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for key, spec in config.MODELS.items():
        rows.append({
            "model": spec["label"],
            "beta_pooled": results["pooled"][key].beta,
            "beta_between": results["between"][key].beta,
            "beta_within": results["within"][key].beta,
            "se_clustered": results["within"][key].se,
            "pvalue_within": results["within"][key].pvalue,
        })
    return pd.DataFrame(rows)


def main() -> None:
    # 1. Data --------------------------------------------------------------
    frames = data_prep.build()
    print(f"Loaded {len(frames['wide'])} country-year rows | "
          f"{frames['wide']['cnt'].nunique()} countries | "
          f"Model A/B sample N = {len(frames['AB'])}")

    # 2. Estimation --------------------------------------------------------
    results = models.run_all(frames)

    _print_header("SUMMARY TABLE - beta by estimator")
    summary = _summary_table(results)
    print(summary.to_string(index=False,
                            float_format=lambda v: f"{v:.4f}"))

    _print_header("RANK PRESERVATION TEST (H0: beta = 1)")
    for key in ("B", "C"):
        t = results["rank"][key]
        print(f"  {t['label']:24} beta = {t['beta']:.4f} | "
              f"clustered SE = {t['se']:.4f} | t = {t['t']:.3f} | p = {t['p']:.2e}")

    _print_header("REDISTRIBUTION REACTION MODEL (between, N=32)")
    rc = results["reaction"]
    print(f"  reduc_public ~ ehearng | beta = {rc['beta']:.4f} | "
          f"R2 = {rc['r2']:.4f} | p = {rc['pvalue']:.2e}")

    # 3. Diagnostics -------------------------------------------------------
    diag = diagnostics.run_all(frames)
    _print_header("DIAGNOSTICS")
    print("Breusch-Pagan (H0: homoskedastic pooled residuals)")
    for key in config.MODELS:
        d = diag["breusch_pagan"][key]
        print(f"  Model {key}: LM = {d['stat']:.3f} | p = {d['pvalue']:.4f}")
    print("Breusch-Godfrey on FE residuals (H0: no serial correlation)")
    for key in config.MODELS:
        d = diag["breusch_godfrey"][key]
        print(f"  Model {key}: LM = {d['stat']:.3f} | p = {d['pvalue']:.2e}")

    # 4. Persist tables and intermediate data ------------------------------
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES_DIR / "summary_beta_by_estimator.csv", index=False)
    frames["between"].to_csv(config.TABLES_DIR / "country_averages.csv", index=False)
    frames["wide"].to_csv(config.TABLES_DIR / "gini_wide.csv", index=False)

    diag_rows = [
        {"model": k, "test": "breusch_pagan", **diag["breusch_pagan"][k]}
        for k in config.MODELS
    ] + [
        {"model": k, "test": "breusch_godfrey_fe", **diag["breusch_godfrey"][k]}
        for k in config.MODELS
    ]
    pd.DataFrame(diag_rows).to_csv(
        config.TABLES_DIR / "diagnostics.csv", index=False)

    # 5. Figures -----------------------------------------------------------
    saved = plots.save_all(frames, results)

    _print_header("OUTPUTS WRITTEN")
    for path in [config.TABLES_DIR / "summary_beta_by_estimator.csv",
                 config.TABLES_DIR / "country_averages.csv",
                 config.TABLES_DIR / "gini_wide.csv",
                 config.TABLES_DIR / "diagnostics.csv", *saved]:
        print(f"  {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
