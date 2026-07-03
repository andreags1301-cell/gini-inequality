"""Project configuration: paths, regions, colours and model specifications.

Centralising these constants keeps the analysis modules free of magic values
and makes the whole pipeline reproducible from a single place.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Resolve everything relative to the repository root so the code runs the same
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "gini_db.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# --- Income concepts (Gini categories in the long-format data) ---------------
# earng    -> Gini of gross earnings (individual labour income)
# ehearng  -> Gini of market income (labour + capital, per capita) = "before"
# ehginc   -> Gini of gross income = market income + public transfers (no taxes)
# ehdinc_  -> Gini of equivalised disposable income = ehginc - taxes = "after"
INCOME_CONCEPTS = ["earng", "ehearng", "ehginc", "ehdinc_"]

# --- Regional groupings (ISO-2 country codes) --------------------------------
REGIONS = {
    "Nordic": ["DK", "FI", "IS", "NO", "SE"],
    "Eastern Europe": ["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "RS", "SK", "SI"],
    "Southern Europe": ["CY", "EL", "ES", "IT", "MT", "PT"],
    "Western Europe": ["AT", "BE", "CH", "DE", "FR", "IE", "LU", "NL", "UK"],
}

REGION_COLOURS = {
    "Nordic": "#2196F3",
    "Western Europe": "#4CAF50",
    "Southern Europe": "#FF9800",
    "Eastern Europe": "#E91E63",
    "Other": "#9E9E9E",
}

# --- Model specifications ----------------------------------------------------
# Each model is a bivariate regression  y = alpha + beta * x  estimated with
# three estimators (pooled OLS, between, within FE). See the research report.
MODELS = {
    "A": {
        "y": "ehdinc_",
        "x": "earng",
        "label": "A: ehdinc_ ~ earng",
        "description": "Hard Left - earnings inequality -> disposable income",
    },
    "B": {
        "y": "ehdinc_",
        "x": "ehearng",
        "label": "B: ehdinc_ ~ ehearng",
        "description": "Social Democrat (main) - market income -> disposable income",
    },
    "C": {
        "y": "ehginc",
        "x": "ehearng",
        "label": "C: ehginc ~ ehearng",
        "description": "Transfers only - market income -> gross income",
    },
}

# Analysis periods used to colour the fixed-effects scatter plot.
PERIODS = [
    ("Crisis (2005-2010)", 2010, "#1f77b4"),
    ("Recovery (2011-2015)", 2015, "#2ca02c"),
    ("Stability (2016-2019)", 2019, "#ffbf00"),
    ("Covid (2020-2023)", 9999, "#d62728"),
]
