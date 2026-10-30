"""Bounded statistical supplement for Fig. 6a, using existing country results.

Country-level inference is conditional on this one simulated scenario. Resampling
countries does not capture dependence through trade or uncertainty in the ABM.
"""
from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

ROOT = bundle_root()
METRICS_SOURCE = data_path('results/postprocessing/formal_abm_metrics/e2eci_with_formal_satisfaction.csv')
SCORES_SOURCE = data_path('results/postprocessing/e2eci/seven_stage_e2eci_country_scores.csv')
OUTPUT_DIR = data_path('results/postprocessing/fig6_statistics')
ENDPOINT_TOLERANCE = 1e-8  # percentage points; same tolerance as the existing figure
SEED = 20260913
BOOTSTRAPS = 20_000
PERMUTATIONS = 99_999


def prepare_data(raw: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Align existing inputs, fail on missing data, and tie numerical endpoints."""
    if not {"country", "E2ECI", "fulfilment_percent"}.issubset(raw):
        raise ValueError("Missing Fig. 6 country metrics")
    if not {"Country", "E2ECI", "final_demand"}.issubset(scores):
        raise ValueError("Missing baseline demand or upstream E2ECI")
    if (raw.country.isna().any() or scores.Country.isna().any()
            or raw.country.duplicated().any() or scores.Country.duplicated().any()):
        raise ValueError("Country identifiers must be present and unique")
    if set(raw.country) != set(scores.Country):
        raise ValueError("Country sets differ; do not silently drop economies")
    data = raw[["country", "E2ECI", "fulfilment_percent"]].rename(
        columns={"fulfilment_percent": "satisfaction_pct"}).merge(
            scores[["Country", "E2ECI", "final_demand"]].rename(
                columns={"Country": "country", "E2ECI": "upstream_E2ECI"}),
            on="country", validate="one_to_one")
    numeric = data[["E2ECI", "upstream_E2ECI", "satisfaction_pct", "final_demand"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (data.final_demand <= 0).any():
        raise ValueError("Inputs must be finite with positive baseline demand")
    if not np.allclose(data.E2ECI, data.upstream_E2ECI, atol=1e-8, rtol=0):
        raise ValueError("E2ECI inputs disagree")
    for column in ("E2ECI", "satisfaction_pct"):
        if ((data[column] < -ENDPOINT_TOLERANCE) | (data[column] > 100 + ENDPOINT_TOLERANCE)).any():
            raise ValueError(f"{column} is outside its 0-100 bounds")
    # Only outcome endpoints are canonicalized; do not round ordinary observations.
    for endpoint in (0., 100.):
        mask = np.abs(data.satisfaction_pct - endpoint) <= ENDPOINT_TOLERANCE
        data.loc[mask, "satisfaction_pct"] = endpoint
    return data.drop(columns="upstream_E2ECI").sort_values("country").reset_index(drop=True)


def load_data() -> pd.DataFrame:
    data = prepare_data(pd.read_csv(METRICS_SOURCE), pd.read_csv(SCORES_SOURCE))
    if len(data) != 168:
        raise ValueError(f"Expected the existing 168-country sample, found {len(data)}")
    return data


def weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be finite and positive")
    weights = weights / weights.sum()
    dx, dy = x - np.sum(weights * x), y - np.sum(weights * y)
    denominator = np.sqrt(np.sum(weights * dx**2) * np.sum(weights * dy**2))
    if denominator <= 0:
        raise ValueError("Correlation is undefined for a constant variable")
    return float(np.sum(weights * dx * dy) / denominator)


def bootstrap_ci(x, y, *, weights=None, rank=False,
                 n_resamples=BOOTSTRAPS, seed=SEED) -> tuple[float, float]:
    """Paired-country percentile interval; weights travel with sampled countries.

    Spearman ranks are recomputed within every bootstrap sample, including ties.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if rank and weights is not None:
        raise ValueError("Weighted rank correlation is not part of this analysis")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, n_resamples, 500):
        indices = rng.integers(0, len(x), size=(min(500, n_resamples - start), len(x)))
        bx, by = x[indices], y[indices]
        if rank:
            bx, by = stats.rankdata(bx, axis=1), stats.rankdata(by, axis=1)
        bw = np.ones_like(bx) if weights is None else np.asarray(weights, float)[indices]
        bw /= bw.sum(axis=1, keepdims=True)
        bx = bx - np.sum(bw * bx, axis=1, keepdims=True)
        by = by - np.sum(bw * by, axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(bw * bx**2, axis=1) * np.sum(bw * by**2, axis=1))
        if (denominator <= 0).any():
            raise ValueError("Undefined correlation in bootstrap; sample is too degenerate")
        estimates.extend(np.sum(bw * bx * by, axis=1) / denominator)
    return tuple(float(v) for v in np.quantile(estimates, [.025, .975]))


def permutation_spearman(x, y, *, n_resamples=PERMUTATIONS, seed=SEED):
    """Two-sided Monte Carlo pairing test, with ties and plus-one correction."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denominator == 0:
        raise ValueError("Spearman correlation needs nonconstant variables")
    estimate = float(rx @ ry / denominator)
    rng = np.random.default_rng(seed)
    extreme = sum(abs(float(rx @ rng.permutation(ry) / denominator)) >= abs(estimate) - 1e-12
                  for _ in range(n_resamples))
    return {"estimate": estimate, "p_value": (extreme + 1) / (n_resamples + 1),
            "extreme_count": int(extreme), "permutations": n_resamples}


def fractional_logit(x, y) -> dict:
    """Bernoulli QML for a fractional outcome, HC1 sandwich standard errors.

    No binary recoding, pseudo-counts, clipping of true endpoints, or log(y) is
    used. The reported coefficient is per 10 E2ECI points on the mean-logit scale.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (len(x) <= 2 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or (y < 0).any() or (y > 1).any() or np.ptp(x) == 0 or np.ptp(y) == 0):
        raise ValueError("Invalid fractional response or design")
    design = np.column_stack([np.ones(len(x)), (x - x.mean()) / 10.])
    def objective(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0., eta) - y * eta))
    def gradient(beta):
        return design.T @ (special.expit(design @ beta) - y)
    fit = optimize.minimize(objective, [special.logit(y.mean()), 0.], jac=gradient,
                            method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    gradient_max = float(np.max(np.abs(gradient(fit.x))))
    if not np.isfinite(fit.fun) or gradient_max > 1e-7:
        raise RuntimeError(f"Fractional logit did not converge: {fit.message}; score={gradient_max}")
    mu = special.expit(design @ fit.x)
    information = design.T @ ((mu * (1 - mu))[:, None] * design)
    bread = np.linalg.inv(information)
    meat = design.T @ (((y - mu)**2)[:, None] * design)
    covariance = bread @ meat @ bread * len(x) / (len(x) - 2)
    se = float(np.sqrt(covariance[1, 1]))
    estimate = float(fit.x[1])
    margin = stats.norm.ppf(.975) * se
    return {"n": len(x), "estimate": estimate, "standard_error": se,
            "ci_low": float(estimate - margin), "ci_high": float(estimate + margin),
            "p_value": float(2 * stats.norm.sf(abs(estimate / se))),
            "intercept": float(fit.x[0]), "x_center": float(x.mean()),
            "score_max_abs": gradient_max, "covariance": "HC1 sandwich",
            "predicted_min": float(mu.min()), "predicted_max": float(mu.max())}


def analyse(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    x, y = data.E2ECI.to_numpy(float), data.satisfaction_pct.to_numpy(float)
    weights = data.final_demand.to_numpy(float)
    rows = []
    def correlation_row(key, label, a, b, *, rank=False, seed_offset=0):
        if rank:
            result = permutation_spearman(a, b, seed=SEED + seed_offset)
            p_method = "Two-sided pairing permutation; plus-one correction"
        else:
            fit = stats.pearsonr(a, b)
            result = {"estimate": float(fit.statistic), "p_value": float(fit.pvalue)}
            p_method = "Conventional two-sided Pearson test (nominal)"
        low, high = bootstrap_ci(a, b, rank=rank, seed=SEED + seed_offset)
        return dict(result, key=key, analysis=label, n=len(a), ci_low=low, ci_high=high,
                    ci_method="95% paired-country percentile bootstrap", p_method=p_method)
    rows.append(correlation_row("pearson", "Equal-weight Pearson r", x, y))
    rows.append(correlation_row("spearman", "Equal-weight Spearman rho", x, y, rank=True, seed_offset=1))
    fractional = fractional_logit(x, y / 100.)
    rows.append(dict(fractional, key="fractional_logit", analysis="Fractional-logit coefficient per 10 E2ECI points",
                     ci_method="95% HC1 sandwich Wald interval", p_method="Two-sided robust Wald test (nominal)"))
    below = y < 100.
    rows.append(correlation_row("below_ceiling_pearson", "Pearson r excluding 100% outcomes", x[below], y[below], seed_offset=2))
    rows.append(correlation_row("below_ceiling_spearman", "Spearman rho excluding 100% outcomes", x[below], y[below], rank=True, seed_offset=3))
    low, high = bootstrap_ci(x, y, weights=weights, seed=SEED + 4)
    rows.append({"key": "demand_weighted", "analysis": "2023 final-demand-weighted Pearson r", "n": len(x),
                 "estimate": weighted_pearson(x, y, weights), "ci_low": low, "ci_high": high,
                 "ci_method": "95% paired-country percentile bootstrap, carrying demand weights",
                 "p_value": None, "p_method": "Not computed; demand weights are not replicate counts"})
    loo = []
    for index, country in enumerate(data.country):
        keep = np.arange(len(x)) != index
        fit = stats.pearsonr(x[keep], y[keep])
        loo.append({"omitted_country": country, "n": int(keep.sum()),
                    "pearson_r": float(fit.statistic), "p_value_nominal": float(fit.pvalue)})
    loo = pd.DataFrame(loo)
    linear = stats.linregress(x, y)
    residuals = y - (linear.intercept + linear.slope * x)
    leverage = 1 / len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2)
    mse = np.sum(residuals**2) / (len(x) - 2)
    cooks = residuals**2 / (2 * mse) * leverage / (1 - leverage)**2
    summary = {
        "n": len(x), "ceiling_count": int((~below).sum()),
        "scenario": "multi-hazard-max / SSP370 / free trade / inv4 / cap1.1 / 2100 relative to 2023",
        "resampling": {"seed": SEED, "bootstrap_samples": BOOTSTRAPS, "permutations": PERMUTATIONS,
                       "endpoint_tolerance_percentage_points": ENDPOINT_TOLERANCE,
                       "unit": "country; deterministic ISO order; offsets 0,1,2,3,4 by table row"},
        "results": {row["key"]: row for row in rows},
        "leave_one_out": {"n_each": len(x) - 1, "r_min": float(loo.pearson_r.min()),
                          "r_max": float(loo.pearson_r.max()),
                          "min_after_omitting": str(loo.loc[loo.pearson_r.idxmin(), "omitted_country"]),
                          "max_after_omitting": str(loo.loc[loo.pearson_r.idxmax(), "omitted_country"]),
                          "largest_nominal_p": float(loo.p_value_nominal.max())},
        "diagnostics": {"ols_residual_skewness": float(stats.skew(residuals, bias=False)),
                        "max_cooks_distance": float(cooks.max()),
                        "max_cooks_country": str(data.country.iloc[int(cooks.argmax())]),
                        "largest_demand_weight_share": float(weights.max() / weights.sum())},
        "interpretation_boundary": "Conditional cross-country associations within one coupled simulation; nominal tests and country resampling do not model trade-network dependence, scenario uncertainty, or causality.",
    }
    return summary, loo


def source_hashes() -> dict[str, str]:
    return {path.relative_to(data_path()).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (METRICS_SOURCE, SCORES_SOURCE)}


def load_verified_summary() -> dict:
    path = data_path('results/postprocessing/fig6_statistics/summary.json')
    if not path.is_file():
        raise ValueError("Run codes/04_postprocessing/04_build_fig6_statistics.py before plotting Fig. 6")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("source_sha256") != source_hashes():
        raise ValueError("Fig. 6 statistics are stale; rebuild from the current inputs")
    return summary
