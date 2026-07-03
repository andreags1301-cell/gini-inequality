# =============================================================================

# ECONOMICS OF INEQUALITY — SEMINAR PAPER v3

# Hard Left vs Social Democrats: Limits of Redistribution

# =============================================================================

# ESTIMATION STRATEGY (mirrors Sections 4 & 5 of the paper):

# 

# STEP 1 — Baseline pooled OLS: y = α + β·X

# Simple bivariate regression, all 523 obs, one intercept.

# Gives the raw relationship. Closest to the course framework.

# 

# STEP 2 — Diagnostics on pooled OLS residuals:

# Breusch-Pagan  → heteroskedasticity?

# Breusch-Godfrey → serial autocorrelation? (expected: yes)

# 

# STEP 3 — Main specification: FE within + clustered SE

# FE neutralises permanent cross-country differences (no observed

# country-level variables needed — demeaning handles everything).

# Clustered SE corrects for confirmed serial autocorrelation.

# 

# Variables:

# earng    → Gini of gross earnings (individual labour income)

# ehearng  → Gini of market income (labour + capital, per capita) = “before”

# ehginc   → Gini of gross income  = ehearng + public transfers (no taxes)

# ehdinc_  → Gini of adjusted disposable income = ehginc − taxes = “after”

# =============================================================================

# — 0. PACKAGES ———————————————————––

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, plm, lmtest, sandwich, ggrepel, patchwork, scales)

# — 1. IMPORT & RESHAPE —————————————————–

gini_db <- read.csv(file.choose(), stringsAsFactors = FALSE) %>%
  mutate(
    year    = as.integer(year),
    gini    = as.numeric(gini),
    cnt     = trimws(cnt),
    inc_cat = trimws(inc_cat)
  ) %>%
  filter(!is.na(gini))

gini_wide <- gini_db %>%
  pivot_wider(names_from = inc_cat, values_from = gini) %>%
  arrange(cnt, year)

glimpse(gini_wide)

# — 2. REGIONS & COLOURS ––––––––––––––––––––––––––

nordic   <- c("DK","FI","IS","NO","SE")
eastern  <- c("BG","CZ","EE","HR","HU","LT","LV","PL","RO","RS","SK","SI")
southern <- c("CY","EL","ES","IT","MT","PT")
western  <- c("AT","BE","CH","DE","FR","IE","LU","NL","UK")

col_region <- c(
  "Nordic"          = "#2196F3",
  "Western Europe"  = "#4CAF50",
  "Southern Europe" = "#FF9800",
  "Eastern Europe"  = "#E91E63",
  "Other"           = "#9E9E9E"
)

gini_wide <- gini_wide %>%
  mutate(region = case_when(
    cnt %in% nordic   ~ "Nordic",
    cnt %in% eastern  ~ "Eastern Europe",
    cnt %in% southern ~ "Southern Europe",
    cnt %in% western  ~ "Western Europe",
    TRUE ~ "Other"
  ))

# — 3. FILTERED DATA ––––––––––––––––––––––––––––

# Models A and B: ehdinc_ ~ earng / ehearng

df_AB <- gini_wide %>%
  filter(!is.na(earng), !is.na(ehearng), !is.na(ehdinc_))

# Model C: ehginc ~ ehearng

df_C <- gini_wide %>%
  filter(!is.na(ehearng), !is.na(ehginc))

# Between (country averages)

df_btw_AB <- df_AB %>%
  group_by(cnt, region) %>%
  summarise(across(c(earng, ehearng, ehginc, ehdinc_), mean, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(
    reduc_total  = earng   - ehdinc_,
    reduc_public = ehearng - ehdinc_,
    reduc_transf = ehearng - ehginc,
    gain_transf  = ehginc  - ehearng
  )

# — 4. PANEL DATA ———————————————————–

panel_AB <- pdata.frame(df_AB, index = c("cnt", "year"))
panel_C  <- pdata.frame(df_C,  index = c("cnt", "year"))

# =============================================================================

# STEP 1 — BASELINE: POOLED OLS  (y = α + β·X)

# =============================================================================

# This is the starting point: the simplest possible model, applied to all 523

# country-year observations jointly. One intercept α, one slope β.

# Closest to the course framework. Gives the raw relationship.

cat("\n", rep("=",70), "\n")
cat("STEP 1 — POOLED OLS BASELINE (y = alpha + beta*X)\n")
cat(rep("=",70), "\n\n")

pool_A <- lm(ehdinc_ ~ earng,   data = df_AB)
pool_B <- lm(ehdinc_ ~ ehearng, data = df_AB)
pool_C <- lm(ehginc  ~ ehearng, data = df_C)

cat("— MODEL A: ehdinc_ ~ earng —\n")
print(summary(pool_A))
cat("HC3 robust SE:\n")
print(coeftest(pool_A, vcov = vcovHC(pool_A, type = "HC3")))

cat("\n— MODEL B: ehdinc_ ~ ehearng (MAIN SPECIFICATION) —\n")
print(summary(pool_B))
cat("HC3 robust SE:\n")
print(coeftest(pool_B, vcov = vcovHC(pool_B, type = "HC3")))

cat("\n— MODEL C: ehginc ~ ehearng —\n")
print(summary(pool_C))
cat("HC3 robust SE:\n")
print(coeftest(pool_C, vcov = vcovHC(pool_C, type = "HC3")))

cat("\n— POOLED OLS SUMMARY —\n")
cat(sprintf(" Model A: β = %.4f | HC3 SE = %.4f | R² = %.3f\n",
             coef(pool_A)[2], sqrt(diag(vcovHC(pool_A, type="HC3")))[2],
             summary(pool_A)$r.squared))
cat(sprintf("  Model B: β = %.4f | HC3 SE = %.4f | R² = %.3f\n",
             coef(pool_B)[2], sqrt(diag(vcovHC(pool_B, type="HC3")))[2],
             summary(pool_B)$r.squared))
cat(sprintf("  Model C: β = %.4f | HC3 SE = %.4f | R² = %.3f\n",
             coef(pool_C)[2], sqrt(diag(vcovHC(pool_C, type="HC3")))[2],
             summary(pool_C)$r.squared))

# =============================================================================

# STEP 2 — DIAGNOSTICS: What the pooled OLS misses

# =============================================================================

# Two tests motivate the move to FE + clustered SE.

cat("\n", rep("=",70), "\n")
cat("STEP 2 — DIAGNOSTICS ON POOLED OLS RESIDUALS\n")
cat(rep("=",70), "\n\n")

cat("— HETEROSKEDASTICITY (Breusch-Pagan) —\n")
cat("H0: residual variance is constant\n\n")
cat("Model A: "); print(bptest(pool_A))
cat("Model B: "); print(bptest(pool_B))
cat("Model C: "); print(bptest(pool_C))

# Note: autocorrelation tested on FE residuals (more meaningful than on pooled)

# We first estimate the FE models here, then test them below

fe_A <- plm(ehdinc_ ~ earng,   data = panel_AB, model = "within")
fe_B <- plm(ehdinc_ ~ ehearng, data = panel_AB, model = "within")
fe_C <- plm(ehginc  ~ ehearng, data = panel_C,  model = "within")

cat("\n— SERIAL AUTOCORRELATION (Breusch-Godfrey on FE residuals) —\n")
cat("H0: errors are uncorrelated over time within countries\n")
cat("Expected result: REJECT — Gini is persistent by nature\n\n")
cat("Model A FE: "); print(pbgtest(fe_A))
cat("Model B FE: "); print(pbgtest(fe_B))
cat("Model C FE: "); print(pbgtest(fe_C))

cat("\n— DIAGNOSTIC CONCLUSIONS —\n")
cat("  Heteroskedasticity: not serious for A and B; borderline for C (BP p=0.046)\n")
cat("  Serial autocorrelation: STRONGLY confirmed in all models (chi²>147, p<2.2e-16)\n")
cat("  → Motivation for FE: remove permanent cross-country heterogeneity\n")
cat("  → Motivation for clustered SE: correct for serial autocorrelation\n")

# =============================================================================

# STEP 3 — MAIN SPECIFICATION: FE within + clustered SE

# =============================================================================

# FE (within estimator): demeans data within each country, so β is identified

# purely from within-country time variation. No country-specific data needed —

# demeaning neutralises ALL permanent cross-country differences at once.

# 

# Clustered SE: allow arbitrary correlation of residuals within the same country

# over time. Effective sample = 32 clusters, not 523 observations.

cat("\n", rep("=",70), "\n")
cat("STEP 3 — MAIN SPECIFICATION: FE WITHIN + CLUSTERED SE\n")
cat(rep("=",70), "\n\n")

# — MODEL A —————————————————————–

cat("=== MODEL A: ehdinc_ ~ earng (Hard Left — total potential) ===\n")
cat("    Does earnings inequality predetermine final inequality?\n\n")
print(summary(fe_A))
cat("\n— Clustered SE —\n")
print(coeftest(fe_A, vcov = vcovHC(fe_A, cluster = "group")))

# — MODEL B —————————————————————–

cat("\n=== MODEL B: ehdinc_ ~ ehearng (Social Democrat — main spec) ===\n")
cat("    What does the full fiscal system (transfers + taxes) do?\n\n")
print(summary(fe_B))
cat("\n— Clustered SE —\n")
print(coeftest(fe_B, vcov = vcovHC(fe_B, cluster = "group")))

# — MODEL C —————————————————————–

cat("\n=== MODEL C: ehginc ~ ehearng (Transfers only) ===\n")
cat("    What do public transfers alone do (before taxes)?\n")
cat("    NOTE: high between R² partly mechanical — see paper Section 3.2\n\n")
print(summary(fe_C))
cat("\n— Clustered SE —\n")
print(coeftest(fe_C, vcov = vcovHC(fe_C, cluster = "group")))

# — RANK PRESERVATION TEST H0: β = 1 ––––––––––––––––––––

cat("\n=== RANK PRESERVATION TEST (H0: β = 1) ===\n")
test_rank <- function(mod, label) {
  b  <- coef(mod)[1]
  se <- sqrt(diag(vcovHC(mod, cluster = "group")))[1]
  t  <- (b - 1) / se
  p  <- 2 * pt(abs(t), df = length(residuals(mod)) - 2, lower.tail = FALSE)
  cat(sprintf("  %s\n  β = %.4f | Clustered SE = %.4f | t(β=1) = %.3f | p = %.6f\n\n",
               label, b, se, t, p))
}
test_rank(fe_B, "Model B (ehdinc_ ~ ehearng)")
test_rank(fe_C, "Model C (ehginc ~ ehearng)")

# =============================================================================

# BETWEEN ESTIMATOR (country averages — structural cross-country comparison)

# =============================================================================

# Note: descriptive pattern, not a causal estimate. Countries differ along

# many structural dimensions simultaneously (institutions, unionisation, etc.)

cat("\n", rep("=",70), "\n")
cat("BETWEEN ESTIMATOR (N=32 country averages)\n")
cat(rep("=",70), "\n\n")

btw_A <- lm(ehdinc_ ~ earng,   data = df_btw_AB)
btw_B <- lm(ehdinc_ ~ ehearng, data = df_btw_AB)
btw_C <- lm(ehginc  ~ ehearng, data = df_btw_AB)

cat(sprintf("  Model A between: β = %.4f | R² = %.3f\n",
             coef(btw_A)[2], summary(btw_A)$r.squared))
cat(sprintf("  Model B between: β = %.4f | R² = %.3f\n",
             coef(btw_B)[2], summary(btw_B)$r.squared))
cat(sprintf("  Model C between: β = %.4f | R² = %.3f (partly mechanical)\n",
             coef(btw_C)[2], summary(btw_C)$r.squared))

# =============================================================================

# ROBUSTNESS: comparison of SE across specifications

# =============================================================================

cat("\n", rep("=",70), "\n")
cat("ROBUSTNESS — COMPARISON OF SE ACROSS SPECIFICATIONS\n")
cat(rep("=",70), "\n\n")

rob <- function(fe_mod, pool_mod, label) {
  b_fe    <- coef(fe_mod)[1]
  se_fe   <- sqrt(diag(vcov(fe_mod)))[1]
  se_cl   <- sqrt(diag(vcovHC(fe_mod, cluster = "group")))[1]
  b_pool  <- coef(pool_mod)[2]
  se_pool <- sqrt(diag(vcov(pool_mod)))[2]
  se_hc3  <- sqrt(diag(vcovHC(pool_mod, type = "HC3")))[2]
  
  cat(sprintf("%s\n", label))
  cat(sprintf("  Pooled OLS  β = %.4f | Std SE = %.4f | HC3 SE = %.4f | t = %.2f\n",
               b_pool, se_pool, se_hc3, b_pool / se_hc3))
  cat(sprintf("  FE within   β = %.4f | Std SE = %.4f | Clust SE = %.4f | t = %.2f\n\n",
               b_fe, se_fe, se_cl, b_fe / se_cl))
}

rob(fe_A, pool_A, "MODEL A: ehdinc_ ~ earng")
rob(fe_B, pool_B, "MODEL B: ehdinc_ ~ ehearng")
rob(fe_C, pool_C, "MODEL C: ehginc  ~ ehearng")

# =============================================================================

# REDISTRIBUTION REACTION MODEL

# =============================================================================

cat("\n", rep("=",70), "\n")
cat("REACTION MODEL: does redistribution respond to pre-tax inequality?\n")
cat(rep("=",70), "\n\n")

mod_reac <- lm(reduc_public ~ ehearng, data = df_btw_AB)
cat("reduc_public = ehearng - ehdinc_ (total Gini reduction via public policy)\n\n")
print(summary(mod_reac))

mod_reac_transf <- lm(gain_transf ~ ehearng, data = df_btw_AB)
cat("\ngain_transf = ehginc - ehearng (transfers only; negative = equalising)\n")
cat("Negative β expected: more unequal countries receive more equalising transfers\n\n")
print(summary(mod_reac_transf))

# =============================================================================

# FINAL SUMMARY TABLE

# =============================================================================

cat("\n", rep("=",70), "\n")
cat("FINAL SUMMARY TABLE\n")
cat(rep("=",70), "\n")
cat(sprintf("%-30s %-10s %-10s %-12s %-10s\n",
             "Model", "β_pool", "β_between", "β_within(FE)", "SE_clust"))
cat(rep("-",75), "\n")

for (r in list(
  list("A: ehdinc_ ~ earng",
        coef(pool_A)[2], coef(btw_A)[2], coef(fe_A)[1],
        sqrt(diag(vcovHC(fe_A, cluster="group")))[1]),
  list("B: ehdinc_ ~ ehearng",
        coef(pool_B)[2], coef(btw_B)[2], coef(fe_B)[1],
        sqrt(diag(vcovHC(fe_B, cluster="group")))[1]),
  list("C: ehginc  ~ ehearng",
        coef(pool_C)[2], coef(btw_C)[2], coef(fe_C)[1],
        sqrt(diag(vcovHC(fe_C, cluster="group")))[1])
)) {
  cat(sprintf("%-30s %-10.4f %-10.4f %-12.4f %-10.4f\n",
               r[[1]], r[[2]], r[[3]], r[[4]], r[[5]]))
}

cat("\nInterpretation of the three-column progression:\n")
cat("  Pooled OLS mixes between + within variation → overestimates β\n")
cat("  Between estimator isolates structural cross-country pattern (descriptive)\n")
cat("  FE within is the main specification: within-country fiscal responsiveness\n")

# =============================================================================

# VISUALISATIONS

# =============================================================================

# –– GRAPH 1: Pooled OLS scatter (baseline, Step 1) ————————
print(
p1 <- ggplot(df_AB, aes(x = ehearng, y = ehdinc_)) +
  geom_point(size = 2, color = "#2166ac", alpha = 0.6) +
             geom_smooth(method = "lm", color = "blue", se = TRUE, linewidth = 0.9) +
               geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                           color = "red", linewidth = 0.8) +
               labs(
                 title    = "Step 1 — Pooled OLS: Gini after vs Gini before redistribution",
                 subtitle = sprintf("y = α + β·X  |  β = %.3f (HC3 SE = %.3f)  |  N = 523 country-year obs.",
                 coef(pool_B)[2],
                 sqrt(diag(vcovHC(pool_B, type="HC3")))[2]),
             x = "Gini market income (ehearng) — before redistribution",
             y = "Gini disposable income (ehdinc_) — after redistribution",
             caption = "Red dashed = β=1 benchmark (no redistribution) | Blue = OLS fit"
  ) +
  theme_minimal(base_size = 12))


# –– GRAPH 2: Fixed Effects scatter (main result, Step 2) ————————

df_FE_plot <- df_AB %>%
  filter(!is.na(ehearng), !is.na(ehdinc_), !is.na(year)) %>%
  mutate(period = case_when(
    year <= 2010 ~ "Crisis (2005–2010)",
    year <= 2015 ~ "Recovery (2011–2015)",
    year <= 2019 ~ "Stability (2016–2019)",
    TRUE         ~ "Covid (2020–2023)"
  )) %>%
  mutate(period = factor(period,
                         levels = c("Crisis (2005–2010)",
                                    "Recovery (2011–2015)",
                                    "Stability (2016–2019)",
                                    "Covid (2020–2023)")))

x_mean <- mean(df_FE_plot$ehearng, na.rm = TRUE)
y_mean <- mean(df_FE_plot$ehdinc_, na.rm = TRUE)

fe_intercept_visual <- y_mean - coef(fe_B)[1] * x_mean
print(
p2_FE <- ggplot(df_FE_plot, aes(x = ehearng, y = ehdinc_, color = period)) +
  geom_point(size = 2.7, alpha = 0.85) +
             geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                         color = "red", linewidth = 0.8) +
               geom_abline(slope = coef(fe_B)[1], intercept = fe_intercept_visual,
                           color = "black", linewidth = 0.9) +
               scale_color_manual(values = c(
                 "Crisis (2005–2010)"    = "#1f77b4",
                 "Recovery (2011–2015)"  = "#2ca02c",
                 "Stability (2016–2019)" = "#ffbf00",
                 "Covid (2020–2023)"     = "#d62728"
               )) +
               labs(
                 title    = "Step 2 — Fixed Effects: Gini after vs Gini before redistribution",
                 subtitle = sprintf("within: y_it = α_i + β·X_it + u_it  |  β = %.3f (clustered SE = %.3f)  |  N = 523 country-year obs.",
                 coef(fe_B)[1],
                 sqrt(diag(vcovHC(fe_B, cluster="group")))[1]),
             x = "Gini market income (ehearng) — before redistribution",
             y = "Gini disposable income (ehdinc_) — after redistribution",
             caption = "Red dashed = β=1 benchmark (no compression) | Black = FE fit | Colours = period",
             color = "Period"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom"))