"""Redraw accepted Figure 6 from six bundled country/model summary CSVs."""
from pathlib import Path
import argparse
import json
import sys
import warnings

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from figure6_statistics import calculate_statistics, ols_band

DEFAULT_DATA=HERE.parents[1]/'data/Figure_6'
COUNTRIES=['CHN','USA','IND','BRA','AUS','CAN','DEU','GBR']
NAMES={'CHN':'China','USA':'United States','IND':'India','BRA':'Brazil',
       'AUS':'Australia','CAN':'Canada','DEU':'Germany','GBR':'United Kingdom'}
EXPECTED_COUNTRIES=frozenset("ABW AGO ALB AND ARE ARG ARM ATG AUS AUT AZE BEL BEN BFA BGD BGR BHR BIH BLR BLZ BOL BRA BRB BRN BWA CAF CAN CHE CHL CHN CIV CMR COD COG COL CPV CRI CUB CYP CZE DEU DJI DMA DNK DOM DZA ECU EGY ESP EST ETH FIN FJI FRA GAB GBR GEO GHA GIN GMB GNQ GRC GRD GTM GUY HKG HND HRV HUN IDN IND IRL IRN IRQ ISL ISR ITA JAM JOR JPN KAZ KEN KGZ KHM KOR KWT LAO LBN LBY LKA LSO LTU LUX LVA MAC MAR MDA MDG MDV MEX MKD MLT MMR MNE MNG MOZ MRT MSR MUS MWI MYS NAM NER NGA NIC NLD NOR NZL OMN PAK PAN PER PHL POL PRK PRT PRY PSE PYF QAT ROU RUS SAU SDN SEN SGP SLV SRB STP SUR SVK SVN SWE SWZ SYC SYR TGO THA TJK TLS TTO TUN TUR TWN TZA UGA UKR URY USA UZB VCT VEN VNM WSM YEM ZAF ZMB ZWE".split())
STAGE_GROUPS={'Mining':('P1',),'Ironmaking':('P2',),
              'Steelmaking and casting':('P3','P4'),'Forming':('P5',),'Manufacturing':('P6',)}
STAGES=list(STAGE_GROUPS)
TRADES=['free-trade','us-cn-decoupling','bloc-confrontation','global-decoupling']
BLUE,ORANGE='#216D98','#C65B20'
INK,MUTED,GRID='#252C31','#66727B','#E5E9EC'
ROW_Y=np.array([0,1,2,3,4.35,5.35,6.35,7.35])
WIDTH_MM,HEIGHT_MM=180.,180.
CAPACITY_BAR_MAX=1.
# The original country-endpoint table stores no 2100 mass for these sparse
# no-delivery rows, while explicitly reporting 0% satisfaction/100% loss. Keep
# their blank mass fields unchanged; do not permit new missing-value locations.
SPARSE_GLOBAL_ZERO_COUNTRIES=frozenset(('ABW ALB AND ARM ATG BEN BFA BLZ BOL BRB BRN BWA CAF CIV CMR CPV CYP DJI DMA DOM EST ETH FJI GAB GMB GRD GUY HKG IRL ISL JAM KGZ KHM LAO LSO LVA MAC MDG MDV MLT MNE MOZ MSR MUS MWI NAM NER NIC PAN PSE PYF SEN STP SUR SWZ SYC TGO TJK TLS TTO VCT WSM ZWE').split())


def complete_keys(frame,columns,expected):
    actual=list(frame[columns].itertuples(index=False,name=None))
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
        raise ValueError('Incomplete, duplicate or unexpected records: '+str(columns))


def finite(frame,columns):
    if frame[columns].isna().any().any() or not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError('Missing or non-finite values: '+str(columns))


def validate_scores(frame):
    fields={'country','dscci','final_demand_satisfaction','final_demand_loss','mfa_final_demand_2023'}
    if set(frame.columns)!=fields or frame.isna().any().any():
        raise ValueError('Invalid capability fields')
    complete_keys(frame,['country'],[(c,) for c in EXPECTED_COUNTRIES])
    finite(frame,list(fields-{'country'}))
    if frame.mfa_final_demand_2023.le(0).any() or frame.dscci.lt(-1e-8).any() or frame.dscci.gt(100+1e-8).any():
        raise ValueError('Invalid demand or DSCCI')
    if frame.final_demand_satisfaction.lt(-1e-8).any() or frame.final_demand_satisfaction.gt(100+1e-8).any():
        raise ValueError('Final-demand satisfaction outside bounds')
    if not np.allclose(frame.final_demand_satisfaction+frame.final_demand_loss,100.,rtol=0,atol=1e-8):
        raise ValueError('Satisfaction and loss disagree')


def read_scores(path):
    frame=pd.read_csv(path)
    validate_scores(frame)
    frame=frame.rename(columns={'country':'Country','dscci':'E2ECI',
                                'final_demand_satisfaction':'satisfaction_pct','final_demand_loss':'loss_pct'})
    for endpoint in [0.,100.]:
        frame.loc[(frame.satisfaction_pct-endpoint).abs().le(1e-8),'satisfaction_pct']=endpoint
    return frame


def validate_endpoint_masses(frame,groups):
    finite(frame,['year','annual_mass','coverage_days','loss_percent','inventory','capacity'])
    if (not frame.year.isin([2023,2100]).all() or not frame.coverage_days.eq(365).all()
            or frame.annual_mass.lt(0).any() or not frame.climate.eq('ssp370').all()
            or not frame.inventory.eq(4).all() or not frame.capacity.eq(1.1).all()):
        raise ValueError('Invalid endpoint calendar, mass or scenario')
    baseline=frame[frame.year.eq(2023)].set_index(groups)
    if baseline.annual_mass.le(0).any():
        raise ValueError('Non-positive scenario-specific baseline')
    keys=pd.MultiIndex.from_frame(frame[groups]) if len(groups)>1 else frame[groups[0]]
    base=(baseline.annual_mass/baseline.coverage_days).reindex(keys).to_numpy()
    expected=100*(1-(frame.annual_mass.to_numpy()/frame.coverage_days.to_numpy())/base)
    if not np.allclose(frame.loss_percent,expected,rtol=0,atol=1e-10):
        raise ValueError('Endpoint loss is inconsistent with mass and own 2023 baseline')


def validate_global_endpoints(frame):
    complete_keys(frame,['trade','year'],[(t,y) for t in TRADES for y in [2023,2100]])
    validate_endpoint_masses(frame,['trade'])


def summarize_random_endpoints(frame):
    expected=[(r,s,y) for r in np.arange(1,20)/20 for s in range(2026081100,2026081110) for y in [2023,2100]]
    complete_keys(frame,['random_ratio','random_seed','year'],expected)
    if not frame.trade.eq('random-disruption').all():
        raise ValueError('Unexpected random-disruption trade setting')
    validate_endpoint_masses(frame,['random_ratio','random_seed'])
    rows=[]
    for ratio,group in frame[frame.year.eq(2100)].sort_values(['random_ratio','random_seed']).groupby('random_ratio'):
        values=group.loss_percent.to_numpy(float)
        half=float(stats.t.ppf(.975,len(values)-1)*stats.sem(values))
        rows.append({'random_ratio':ratio,'n':len(values),'mean_loss_percent':float(values.mean()),
                     'sd':float(values.std(ddof=1)),'ci95_low':float(values.mean()-half),
                     'ci95_high':float(values.mean()+half)})
    return pd.DataFrame(rows)


def validate_country_endpoints(frame,countries):
    expected=[(c,t) for t in TRADES for c in countries if t!='global-decoupling' or c not in ['LTU','MKD']]
    complete_keys(frame,['country','trade'],expected)
    finite(frame,['2023','fulfilment_percent','loss_percent'])
    sparse=frame['2100'].isna()
    expected_sparse=frame.trade.eq('global-decoupling')&frame.country.isin(SPARSE_GLOBAL_ZERO_COUNTRIES)
    if not np.array_equal(sparse,expected_sparse):
        raise ValueError('Unexpected missing country endpoint mass')
    if not (frame.loc[sparse,'fulfilment_percent'].eq(0).all() and frame.loc[sparse,'loss_percent'].eq(100).all()):
        raise ValueError('Original sparse no-delivery rows disagree with recorded outcomes')
    finite(frame.loc[~sparse],['2100'])
    if frame['2023'].le(0).any() or frame['2100'].lt(0).any():
        raise ValueError('Invalid national endpoint mass')
    known=frame.loc[~sparse]
    if not np.allclose(known.fulfilment_percent,100*known['2100']/known['2023'],rtol=0,atol=1e-8):
        raise ValueError('National endpoint ratio disagrees with masses')
    if not np.allclose(frame.fulfilment_percent+frame.loss_percent,100.,rtol=0,atol=1e-8):
        raise ValueError('National satisfaction/loss mismatch')


def validate_capacity_components(frame,scores):
    expected=[(c,f'P{i}') for c in COUNTRIES for i in range(1,7)]
    complete_keys(frame,['country','sector'],expected)
    fields=['total_inflow','total_outflow','baseline_throughput','capacity_factor','modeled_capacity','final_demand']
    finite(frame,fields)
    if frame[['total_inflow','total_outflow','modeled_capacity']].lt(0).any().any() or frame.final_demand.le(0).any():
        raise ValueError('Invalid capacity components or denominator')
    throughput=np.maximum(frame.total_inflow,frame.total_outflow)
    if (not frame.capacity_factor.eq(1.1).all()
            or not np.allclose(frame.baseline_throughput,throughput,rtol=1e-12,atol=1e-6)
            or not np.allclose(frame.modeled_capacity,throughput*1.1,rtol=1e-12,atol=1e-6)):
        raise ValueError('Model capacity does not match 1.1 times baseline throughput')
    demand=frame.country.map(scores.set_index('country').mfa_final_demand_2023)
    if not np.allclose(frame.final_demand,demand,rtol=1e-12,atol=1e-6):
        raise ValueError('Capacity denominator differs from domestic 2023 MFA final demand')


def load_inputs(data_dir=DEFAULT_DATA):
    folder=Path(data_dir)
    raw_scores=pd.read_csv(folder/'country_capability.csv')
    validate_scores(raw_scores)
    scores=read_scores(folder/'country_capability.csv')
    random=summarize_random_endpoints(pd.read_csv(folder/'random_trade_endpoints.csv'))
    global_endpoints=pd.read_csv(folder/'global_trade_endpoints.csv')
    validate_global_endpoints(global_endpoints)
    named=global_endpoints[global_endpoints.year.eq(2100)].set_index('trade').loc[TRADES].reset_index()
    countries=pd.read_csv(folder/'country_trade_endpoints.csv')
    validate_country_endpoints(countries,raw_scores.country.tolist())
    sums=countries.groupby('trade')[['2023','2100']].sum()
    for year in [2023,2100]:
        global_mass=global_endpoints[global_endpoints.year.eq(year)].set_index('trade').loc[TRADES,'annual_mass']
        if not np.allclose(global_mass,sums.loc[TRADES,str(year)],rtol=1e-10,atol=.001):
            raise ValueError('Global endpoints disagree with summed country masses')
    final_profiles=scores[scores.Country.isin(COUNTRIES)][['Country','satisfaction_pct']].rename(columns={'satisfaction_pct':'Fulfilment'})
    final_profiles['Stage']='Final demand'
    comparison=prepare_comparison(countries,scores,final_profiles)
    capacities=pd.read_csv(folder/'baseline_capacity_components.csv')
    validate_capacity_components(capacities,raw_scores)
    capacities=capacities.rename(columns={'country':'Country','sector':'stage_code'})
    annual=pd.read_csv(folder/'production_components.csv')
    expected=[(row.Country,row.stage_code,year) for row in capacities.itertuples() if row.modeled_capacity>0 for year in [2023,2100]]
    complete_keys(annual,['country','sector','year'],expected)
    production,bars=prepare_production_profiles(annual,capacities)
    summary=calculate_statistics(scores)
    return {'data':(scores,final_profiles,comparison,random,named,summary),
            'annual':annual,'capacity_components':capacities,
            'production_profiles':production,'capacity':bars}

def prepare_comparison(named: pd.DataFrame, scores: pd.DataFrame,
                       profiles: pd.DataFrame) -> pd.DataFrame:
    """Fail on missing/duplicate endpoints or unlike denominators; never fill 0."""
    selected = named.loc[named.country.isin(COUNTRIES)
                         & named.trade.isin(["free-trade", "bloc-confrontation"])].copy()
    if selected.duplicated(["country", "trade"]).any():
        raise ValueError("Duplicate country-scenario endpoints")
    expected = pd.MultiIndex.from_product(
        [COUNTRIES, ["free-trade", "bloc-confrontation"]], names=["country", "trade"])
    indexed = selected.set_index(["country", "trade"])
    if set(indexed.index) != set(expected):
        raise ValueError("Missing country-scenario endpoints; do not replace them with zero")
    selected = indexed.reindex(expected).reset_index()
    numeric = selected[["2023", "2100", "fulfilment_percent", "loss_percent"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (selected["2023"] <= 0).any():
        raise ValueError("Invalid or non-positive baseline")
    if not np.allclose(selected.fulfilment_percent,
                       100 * selected["2100"] / selected["2023"], atol=1e-8, rtol=0):
        raise ValueError("Final-demand ratio disagrees with the source masses")
    if not np.allclose(selected.fulfilment_percent + selected.loss_percent, 100,
                       atol=1e-8, rtol=0):
        raise ValueError("Satisfaction and loss do not sum to 100")
    wide = selected.pivot(index="country", columns="trade").reindex(COUNTRIES)
    free_base = wide["2023"]["free-trade"]
    group_base = wide["2023"]["bloc-confrontation"]
    if not np.allclose(free_base, group_base, atol=0.001, rtol=1e-10):
        raise ValueError("Country comparison has unlike 2023 denominators")
    result = pd.DataFrame({
        "country": COUNTRIES,
        "country_name": [NAMES[c] for c in COUNTRIES],
        "dscci": scores.set_index("Country").loc[COUNTRIES, "E2ECI"].to_numpy(),
        "baseline_2023_free": free_base.to_numpy(),
        "baseline_2023_group": group_base.to_numpy(),
        "free_trade_satisfaction": wide["fulfilment_percent"]["free-trade"].to_numpy(),
        "group_restriction_satisfaction": wide["fulfilment_percent"]["bloc-confrontation"].to_numpy(),
    })
    final_stage = profiles.loc[profiles.Stage.eq("Final demand")].set_index("Country")
    if not np.allclose(result.free_trade_satisfaction,
                       final_stage.loc[COUNTRIES, "Fulfilment"], atol=1e-8, rtol=0):
        raise ValueError("Free-trade endpoints and existing stage profiles disagree")
    for col in ["free_trade_satisfaction", "group_restriction_satisfaction"]:
        for endpoint in [0., 100.]:
            result.loc[np.abs(result[col] - endpoint) <= 1e-8, col] = endpoint
    result["change_percentage_points"] = (
        result.group_restriction_satisfaction - result.free_trade_satisfaction)
    return result


def prepare_production_profiles(annual, source_capacity, groups=STAGE_GROUPS,
                               countries=COUNTRIES):
    """Aggregate parallel branches in mass units before forming relative values.

    Production means actual main-product output, not replenishment receipts.
    Absent agents can contribute zero only when canonical model capacity is zero.
    """
    keys = ["country", "sector", "year"]
    if annual.duplicated(keys).any():
        raise ValueError("Duplicate production records")
    if (not np.isfinite(annual.actual_output).all()
            or annual.actual_output.lt(0).any()):
        raise ValueError("Invalid production mass")
    if source_capacity.duplicated(["Country", "stage_code"]).any():
        raise ValueError("Duplicate capacity records")
    masses = annual.set_index(keys).actual_output
    caps = source_capacity.set_index(["Country", "stage_code"])
    profiles, bars = [], []
    for country in countries:
        for stage, members in groups.items():
            component_keys = [(country, p) for p in members]
            if any(key not in caps.index for key in component_keys):
                raise ValueError(f"Missing capacity components: {country} {stage}")
            component = caps.loc[component_keys]
            demand = component.final_demand.to_numpy(float)
            capacity = component.modeled_capacity.to_numpy(float)
            if (not np.isfinite(demand).all() or (demand <= 0).any()
                    or not np.allclose(demand, demand[0], rtol=1e-12, atol=1e-6)):
                raise ValueError("Invalid or inconsistent final-demand denominator")
            if not np.isfinite(capacity).all() or (capacity < 0).any():
                raise ValueError("Invalid capacity")
            totals = {}
            for year in (2023, 2100):
                values = []
                for sector in members:
                    key = (country, sector, year)
                    if key in masses.index:
                        values.append(float(masses.loc[key]))
                    elif caps.loc[(country, sector), "modeled_capacity"] == 0:
                        values.append(0.)
                    else:
                        raise ValueError(f"Missing production for positive-capacity component: {key}")
                totals[year] = sum(values)
            if totals[2023] == 0 and totals[2100] != 0:
                raise ValueError("Nonzero future production without a baseline")
            ratio = 100 * totals[2100] / totals[2023] if totals[2023] > 0 else np.nan
            profiles.append({
                "Country": country, "Stage": stage, "stage_code": "+".join(members),
                "production_2023": totals[2023], "production_2100": totals[2100],
                "Fulfilment": ratio,
            })
            bars.append({
                "Country": country, "Stage": stage, "stage_code": "+".join(members),
                "modeled_capacity": float(capacity.sum()), "final_demand": demand[0],
                "capacity_to_final_demand": float(capacity.sum() / demand[0]),
            })
    return pd.DataFrame(profiles), pd.DataFrame(bars)


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM])


def text_mm(fig, x, y, text, **kwargs):
    return fig.text(x / WIDTH_MM, y / HEIGHT_MM, text, **kwargs)


def panel_title(fig, letter, x, y, title):
    text_mm(fig, x - 6, y, letter, fontsize=9, weight="bold", va="center")
    text_mm(fig, x, y, title, fontsize=7.4, weight="bold", va="center")


def style_axis(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#919BA2")
    ax.tick_params(length=2.3, width=0.45, color="#919BA2", pad=2.2)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.45)


def draw_scatter(fig, scores, summary):
    ax = axes_mm(fig, 14, 108, 65, 61)
    panel_title(fig, "a", 14, 176, "Domestic capability and steel security")
    x, y = scores.E2ECI.to_numpy(), scores.satisfaction_pct.to_numpy()
    grid = np.linspace(0, 100, 300)
    fitted, low, high, _ = ols_band(x, y, grid)
    ax.fill_between(grid, low, high, color="#DCEAF1", lw=0, zorder=1)
    ax.plot(grid, fitted, color=BLUE, lw=1.0, zorder=2)
    ordinary = ~scores.Country.isin(COUNTRIES)
    ax.scatter(x[ordinary], y[ordinary], s=10, color="#A9B0B5", alpha=.62,
               edgecolors="none", zorder=3)
    labels = {
        "CHN": (62, 103, "right"), "IND": (76, 108, "right"),
        "USA": (96, 104.3, "right"), "BRA": (101, 108, "right"),
        "AUS": (61, 85.7, "left"), "CAN": (30, 86, "left"),
        "DEU": (35, 91.4, "left"), "GBR": (34, 75.5, "left"),
    }
    for country in COUNTRIES:
        row = scores.set_index("Country").loc[country]
        ax.scatter(row.E2ECI, row.satisfaction_pct, s=25, color=BLUE,
                   edgecolors="white", linewidth=.45, zorder=6)
        lx, ly, ha = labels[country]
        ax.annotate(NAMES[country], (row.E2ECI, row.satisfaction_pct),
                    xytext=(lx, ly), ha=ha, va="center", fontsize=5.7,
                    color=INK, zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.6, alpha=.92),
                    arrowprops=dict(arrowstyle="-", color="#7D939F", lw=.45,
                                    shrinkA=2, shrinkB=3))
    ax.axhline(100, color="#B3BDC3", lw=.6, linestyle=(0, (3, 2)), zorder=0)
    ax.set(xlim=(-2, 103), ylim=(65, 111), xticks=np.arange(0, 101, 20),
           yticks=[70, 80, 90, 100], xlabel="Domestic supply chain capability index (DSCCI)",
           ylabel="Final-demand satisfaction (%)")
    ax.xaxis.label.set_size(6.1)
    pearson = summary["results"]["pearson"]
    spearman = summary["results"]["spearman"]
    def pstring(v):
        c, e = f"{v:.1e}".split("e")
        return rf"{c}\times10^{{{int(e)}}}"
    ax.text(.98, .025,
            rf"$n=168$; Pearson $r={pearson['estimate']:.3f}$" + "\n"
            + f"95% CI {pearson['ci_low']:.3f}–{pearson['ci_high']:.3f}; "
            + rf"$P={pstring(pearson['p_value'])}$" + "\n"
            + rf"Spearman $\rho={spearman['estimate']:.3f}$; $P_{{perm}}={pstring(spearman['p_value'])}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3, color=MUTED)
    style_axis(ax)
    return ax


def draw_trade(fig, random, named):
    ax = axes_mm(fig, 14, 59, 65, 21)
    panel_title(fig, "b", 14, 87, "Random trade restrictions")
    x = random.random_ratio.to_numpy() * 100
    mean = random.mean_loss_percent.to_numpy()
    ax.fill_between(x, random.ci95_low, random.ci95_high, color="#DCEAF1", linewidth=0)
    ax.errorbar(x, mean, yerr=np.vstack([mean - random.ci95_low,
                                       random.ci95_high - mean]),
                fmt="o-", ms=2.5, lw=.85, elinewidth=.5, capsize=1.2,
                color=BLUE, markeredgecolor="white", markeredgewidth=.25, zorder=3)
    free_loss = float(named.loc[named.trade.eq("free-trade"), "loss_percent"].iloc[0])
    ax.axhline(free_loss, color="#A4AEB5", ls=(0, (3, 2)), lw=.55)
    ax.text(.98, .20, "Free-trade baseline", transform=ax.transAxes,
            ha="right", fontsize=5.2, color=MUTED)
    ax.text(.03, .93, "Mean ± 95% CI; 10 runs per level", transform=ax.transAxes,
            fontsize=5.4, va="top", color=MUTED)
    ax.set(xlim=(0, 100), ylim=(0, 50), xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50],
           xlabel="Potential foreign trading partners excluded (%)",
           ylabel="Global loss (%)")
    ax.xaxis.label.set_size(5.9)
    style_axis(ax)

    # Leave an intentional gap after panel b's x-axis label. The c title must
    # visually belong to c rather than attach to b's plot/label block.
    ax2 = axes_mm(fig, 14, 14, 65, 22)
    panel_title(fig, "c", 14, 41, "Trade-restriction scenarios")
    vals = named.loss_percent.to_numpy()
    bars = ax2.bar(np.arange(4), vals, width=.48,
                   color=[BLUE, "#91A7B5", ORANGE, "#87543A"], zorder=3)
    for bar, value in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 1.6, f"{value:.1f}",
                 ha="center", va="bottom", fontsize=6.2, color=INK)
    ax2.set(ylim=(0, 54), yticks=[0, 25, 50], xlim=(-.6, 3.6),
            xticks=np.arange(4), ylabel="Global loss (%)")
    ax2.set_xticklabels(["Free\ntrade", "US–China\ndecoupling", "Group\nrestriction",
                         "Global\ndecoupling*"], fontsize=5.8)
    style_axis(ax2)
    ax2.tick_params(axis="x", length=0, pad=3)
    return ax, ax2


def draw_country_panels(fig, profiles, comparison, capacity):
    # Five display groups; P3 and P4 are parallel branches summed before ratios.
    panel_title(fig, "d", 94, 176, "Baseline capacity and production response")
    panel_title(fig, "e", 94, 87, "Final-demand satisfaction")
    text_mm(fig, 94, 171.5, "Free trade", fontsize=5.4, color=MUTED)
    ax = axes_mm(fig, 122, 121, 52, 46)
    ep = axes_mm(fig, 123, 20, 51, 54)
    cmap = LinearSegmentedColormap.from_list("fulfilment", ["#C46028", "#FAFAF8", "#357FA4"])
    norm = Normalize(50, 150)
    matrix = profiles.pivot(index="Country", columns="Stage", values="Fulfilment").loc[COUNTRIES, STAGES]
    ratios = capacity.pivot(index="Country", columns="Stage", values="capacity_to_final_demand").loc[COUNTRIES, STAGES]
    scores = comparison.set_index("country").dscci
    for r, country in enumerate(COUNTRIES):
        y = ROW_Y[r]
        for c, stage in enumerate(STAGES):
            val = float(matrix.loc[country, stage])
            missing = not np.isfinite(val)
            patch = Rectangle((c - .48, y - .46), .96, .92,
                              facecolor="#E6EAED" if missing else cmap(norm(val)),
                              edgecolor="white", lw=.35, hatch="///" if missing else None)
            ax.add_patch(patch)
            ax.text(c, y - .10, "N/A" if missing else f"{val:.1f}",
                    ha="center", va="center", fontsize=5.6,
                    color="white" if not missing and val < 65 else INK)
            # A neutral lower strip separates baseline capacity from the
            # coloured dynamic response. All bars share a linear 0--1x scale.
            ax.add_patch(Rectangle((c - .46, y + .25), .92, .18,
                                   facecolor="#FFFFFF", edgecolor="none", alpha=.84))
            ratio = float(ratios.loc[country, stage])
            left, span, line_y = c - .36, .72, y + .355
            ax.plot([left, left + span], [line_y, line_y], color="#CFD5D9",
                    lw=.65, solid_capstyle="butt")
            length = span * min(ratio, CAPACITY_BAR_MAX) / CAPACITY_BAR_MAX
            if ratio > 0:
                ax.plot([left, left + length], [line_y, line_y], color="#465560",
                        lw=1.05, solid_capstyle="butt")
            if ratio > CAPACITY_BAR_MAX:
                ax.plot(left + span, line_y, marker=">", markersize=2.4,
                        markeredgewidth=0, color="#465560")
        ax.text(-3.5/6, y, NAMES[country], transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.3,
                clip_on=False)
        ax.text(-.4/6, y, f"{scores[country]:.1f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=6.0, color=MUTED, clip_on=False)
    ax.set(xlim=(-.5, len(STAGES)-.5), ylim=(7.9, -.55), xticks=np.arange(len(STAGES)), yticks=[])
    ax.set_xticklabels([s.replace("Steelmaking and casting", "Steelmaking\nand casting") for s in STAGES],
                       rotation=43, ha="right", rotation_mode="anchor", fontsize=5.9)
    ax.tick_params(axis="x", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-3.5/6, -.73, "Country", transform=ax.get_yaxis_transform(),
            fontsize=5.7, color=MUTED, ha="left", va="bottom", clip_on=False)
    ax.text(-.4/6, -.73, "DSCCI", transform=ax.get_yaxis_transform(),
            fontsize=5.7, color=MUTED, ha="right", va="bottom", clip_on=False)
    ax.axhline(3.68, color=GRID, lw=.6, xmin=0, xmax=1)

    for r, row in enumerate(comparison.itertuples(index=False)):
        y = ROW_Y[r]
        # Once d and e no longer share a row, e must be independently readable.
        ep.text(-.55, y, NAMES[row.country], transform=ep.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.3, clip_on=False)
        ep.text(-.065, y, f"{row.dscci:.1f}", transform=ep.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.0, color=MUTED, clip_on=False)
        free, group = row.free_trade_satisfaction, row.group_restriction_satisfaction
        ep.axhline(y, color="#EEF1F3", lw=.5, zorder=0)
        ep.plot([group, free], [y, y], color="#A9B3B9", lw=1.2, zorder=1)
        # Both markers stay visible when the outcomes coincide at 100%.
        ep.scatter(group, y, marker="D", s=26, facecolor="white", edgecolor=ORANGE, lw=.9, zorder=3)
        ep.scatter(free, y, marker="o", s=12, facecolor=BLUE, edgecolor="white", lw=.3, zorder=4)
        if abs(free - group) < 1e-8:
            ep.annotate(f"{free:.1f}", (free, y), xytext=(-5, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=6.2, color=INK)
        else:
            for value, color in [(free, BLUE), (group, ORANGE)]:
                ep.annotate(f"{value:.1f}", (value, y), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.0, color=color)
    ep.set(xlim=(54, 104), ylim=(7.9, -.55), xticks=[60, 80, 100], yticks=[],
           xlabel="Final-demand satisfaction (%)")
    ep.xaxis.label.set_size(6.1)
    ep.axvline(100, color="#B5BEC4", linestyle=(0, (2, 2)), lw=.55, zorder=0)
    ep.axhline(3.68, color=GRID, lw=.6)
    style_axis(ep, grid=None)
    ep.spines["left"].set_visible(False)
    ep.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, markersize=3.3, label="Free trade"),
        Line2D([], [], marker="D", color=ORANGE, markerfacecolor="white", lw=0,
               markersize=3.8, label="Group restriction"),
    ], loc="lower left", bbox_to_anchor=(-.55, 1.115), frameon=False, fontsize=5.5,
              ncol=2, handletextpad=.35, columnspacing=.8, borderaxespad=0)
    text_mm(fig, 94.95, 77.2, "Country", fontsize=5.7, color=MUTED)
    text_mm(fig, 119.685, 77.2, "DSCCI", fontsize=5.7, color=MUTED, ha="right")
    cbax = axes_mm(fig, 122, 103, 52, 2)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax,
                     orientation="horizontal", ticks=[50, 75, 100, 125, 150])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.1, length=1.8, width=.4, pad=1)
    cb.set_label("Production relative to 2023 (%)", size=5.8, labelpad=1.5)
    # Compact bar key replaces the external N/A note. N/A remains in the cells.
    # Keep the common scale and overflow explicit after removing cell ratios.
    text_mm(fig, 94, 114, "Capacity bar", fontsize=5.3, color=MUTED, va="center")
    key = axes_mm(fig, 94, 109.5, 16, 1)
    key.plot([0, 1], [.5, .5], color="#CFD5D9", lw=1.1, solid_capstyle="butt")
    key.plot([0, .5], [.5, .5], color="#465560", lw=1.1, solid_capstyle="butt")
    key.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[])
    key.set_xticklabels(["0", "0.5", "1×"], fontsize=4.8)
    key.tick_params(axis="x", length=1.4, width=.35, pad=.6, color=MUTED)
    for spine in key.spines.values():
        spine.set_visible(False)
    fig.add_artist(Line2D([112.5 / WIDTH_MM], [110 / HEIGHT_MM],
                         transform=fig.transFigure, marker=">", markersize=3.2,
                         markeredgewidth=0, linestyle="none", color=MUTED))
    text_mm(fig, 114, 110, ">1×", fontsize=5.1, color=MUTED, va="center")
    text_mm(fig, 94, 103.5, "Baseline capacity /\n2023 final demand", fontsize=5.3,
            color=MUTED, va="top", linespacing=1.3)
    text_mm(fig, 94, 9,
            "Group restriction: US-aligned–China-partner confrontation.  *Scenario-specific 2023 baseline.",
            fontsize=5.2, color=MUTED, va="top", linespacing=1.4)
    return ax, ep


def build_figure(data, capacity, production_profiles):
    scores, profiles, comparison, random, named, summary = data
    with mpl.rc_context({
        "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": .5,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        # Preserve adopted v6 typography even if an optional imported style
        # changes its formatter or mathematical font defaults.
        "axes.formatter.use_mathtext": False, "mathtext.fontset": "dejavusans",
        # The old project import applied these defaults as a side effect.
        # State them locally instead of importing that module/SciencePlots.
        "lines.linewidth": .9, "lines.markersize": 3., "lines.solid_capstyle": "round",
        "patch.linewidth": 0.,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }):
        fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))
        draw_scatter(fig, scores, summary)
        draw_trade(fig, random, named)
        draw_country_panels(fig, production_profiles, comparison, capacity)
        return fig



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-dir',type=Path,default=HERE/'output')
    args=parser.parse_args()
    try:
        font_manager.findfont('Arial',fallback_to_default=False)
    except ValueError:
        warnings.warn('Arial is unavailable; fallback typography can differ from the reference.',stacklevel=2)
    bundle=load_inputs(args.data_dir)
    fig=build_figure(bundle['data'],bundle['capacity'],bundle['production_profiles'])
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with mpl.rc_context({'savefig.bbox':None,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        for ext,options in [('png',{'dpi':600}),('pdf',{}),('svg',{})]:
            fig.savefig(output/f'Figure_6.{ext}',facecolor='white',**options)
        fig.savefig(output/'Figure_6_preview.png',dpi=240,facecolor='white')
    plt.close(fig)
    scores,_,comparison,random,named,summary=bundle['data']
    for name,frame in [('scatter',scores),('stage_fulfilment',bundle['production_profiles']),
                       ('production_components_annual',bundle['annual']),
                       ('baseline_capacity_components',bundle['capacity_components']),
                       ('baseline_capacity',bundle['capacity']),('country_comparison',comparison),
                       ('random_restrictions',random),('named_scenarios',named)]:
        frame.to_csv(output/f'source_{name}.csv',index=False)
    grid=np.linspace(0,100,300)
    fit,low,high,_=ols_band(scores.E2ECI.to_numpy(),scores.satisfaction_pct.to_numpy(),grid)
    pd.DataFrame({'dscci':grid,'fitted':fit,'low':low,'high':high}).to_csv(output/'source_ols_band.csv',index=False)
    (output/'Figure_6_statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'economies':len(scores),'capacity_bars':len(bundle['capacity']),
                      'finite_production_ratios':int(bundle['production_profiles'].Fulfilment.notna().sum()),
                      'random_levels':len(random)},indent=2))


if __name__=='__main__':
    main()

