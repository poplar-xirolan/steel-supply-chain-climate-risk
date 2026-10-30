"""Step 4: supplement Fig. 6a statistics; reads small existing country tables only."""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = bundle_root()
sys.path.insert(0, str(ROOT))
from steel_supply.postprocessing.fig6_statistics import (
    OUTPUT_DIR, analyse, load_data, source_hashes,
)


def si_table(summary):
    ceiling_count = summary["ceiling_count"]
    def p_text(value):
        if value is None:
            return "—"
        coefficient, exponent = f"{value:.1e}".split("e")
        return f"{coefficient} × 10<sup>{int(exponent)}</sup>"
    labels = {
        "pearson": ("Equal-weight primary analysis", "Pearson r"),
        "spearman": ("Rank-based analysis", "Spearman ρ"),
        "fractional_logit": ("Bounded-response model", "Fractional-logit β per 10 E2ECI points"),
        "below_ceiling_pearson": (f"Exclude the {ceiling_count} economies at 100.0%", "Pearson r"),
        "below_ceiling_spearman": (f"Exclude the {ceiling_count} economies at 100.0%", "Spearman ρ"),
        "demand_weighted": ("Weight by 2023 final demand", "Weighted Pearson r"),
    }
    lines = [
        "# Table S4 | Statistical support and sensitivity checks for the E2ECI–final-demand-satisfaction association in Fig. 6a.",
        "",
        "| Analysis | n | Statistic | Estimate | 95% confidence interval | Two-sided P |",
        "|---|---:|---|---:|---|---|",
    ]
    for key, row in summary["results"].items():
        label, statistic = labels[key]
        suffix = " (permutation)" if "spearman" in key else ""
        lines.append(f"| {label} | {row['n']} | {statistic} | {row['estimate']:.3f} | "
                     f"{row['ci_low']:.3f}–{row['ci_high']:.3f} | {p_text(row['p_value'])}{suffix} |")
    loo = summary["leave_one_out"]
    lines.append(f"| Leave one economy out in turn | {loo['n_each']} per fit | Pearson r range | "
                 f"{loo['r_min']:.3f}–{loo['r_max']:.3f} | — | — |")
    lines.extend([
        "",
        "All analyses use the multi-hazard SSP3-7.0 condition in 2100, free trade, a four-week inventory target, "
        "a 10% production-capacity margin and the fixed 2023 industrial structure. The primary analysis weights "
        f"each of the {summary['n']} economies equally. Values within 10<sup>−8</sup> percentage points of the satisfaction "
        "endpoints were treated as endpoint ties, without rounding other observations or modifying the source data.",
        "",
        "Correlation confidence intervals are percentile intervals from 20,000 paired-country bootstrap samples. "
        "Spearman ranks were recomputed within each sample. Pearson P values are conventional two-sided values; "
        "Spearman P values use 99,999 random permutations of country pairings and the (b + 1)/(B + 1) correction. "
        f"The numbers of permutations at least as extreme as the observed statistic were "
        f"{summary['results']['spearman']['extreme_count']} for the primary Spearman analysis and "
        f"{summary['results']['below_ceiling_spearman']['extreme_count']} after excluding ceiling observations. "
        "The minimum reportable Monte Carlo P value is 0.00001; it is not an exact tail probability. "
        "Random seeds are fixed in the code.",
        "",
        "The fractional-logit model uses final-demand satisfaction divided by 100 as the fractional response, "
        "retaining observations at 1, and E2ECI as the predictor. Its coefficient is per 10 E2ECI points on the "
        "logit scale of the conditional mean, not a percentage-point effect on satisfaction. Estimation uses "
        "Bernoulli quasi-maximum likelihood with HC1 sandwich standard errors and normal-based Wald inference. "
        "See [Papke and Wooldridge (1996)](https://www.jstor.org/stable/2285155).",
        "",
        "Demand weighting uses the existing 2023 final-demand values; each sampled country retains its weight "
        "in the bootstrap. It changes the emphasis toward larger-demand economies and does not replace the "
        "equal-weight primary analysis. No weighted P value is reported because demand weights are not "
        "independent replicate counts. The leave-one-out range is not a confidence interval; the minimum "
        f"occurs after omitting {loo['min_after_omitting']} and the maximum after omitting {loo['max_after_omitting']}.",
        "",
        "These are conditional cross-country associations within one coupled simulation. Nominal tests and "
        "country resampling do not account for trade-network dependence or uncertainty across climate scenarios "
        "and model specifications, and do not establish causality.",
        "",
    ])
    return "\n".join(lines)


def main():
    data = load_data()
    summary, leave_one_out = analyse(data)
    summary["source_sha256"] = source_hashes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["key", "analysis", "n", "estimate", "ci_low", "ci_high", "p_value", "ci_method", "p_method"]
    pd.DataFrame(summary["results"].values())[columns].to_csv(OUTPUT_DIR / "statistics.csv", index=False)
    leave_one_out.to_csv(OUTPUT_DIR / "leave_one_out.csv", index=False)
    (OUTPUT_DIR / "table_S4_fig6_statistics.md").write_text(si_table(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
