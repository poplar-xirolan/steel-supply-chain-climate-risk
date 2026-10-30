"""Redraw Figure 4 from all country/year/half-year absolute loss totals."""
from pathlib import Path
import argparse
import calendar
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common.figure_style import apply_figure_style, check_reference_font, panel_label

DEFAULT_DATA=HERE.parents[1]/'data/Figure_4/country_seasonal_losses.csv'
STAGES=['iron-mine','iron-factory','steel-factory']
HAZARDS=['heatwave','tropical-cyclone','wildfire']
SEASONS=['Apr-Sep','Oct-Mar']
YEARS=set(range(2055,2066))
# Fixed country coverage of the accepted Figure 4 source data. Deriving this
# solely from the input would fail to detect a country whose every row is lost.
EXPECTED_COUNTRIES={
    'iron-mine':frozenset(('AUS AUT BIH BRA CAN CHL CHN DZA IND IRN KAZ LBR MEX MRT '
                           'MYS NOR NZL PER PRK RUS SLE SWE TUR UKR USA VEN ZAF').split()),
    'iron-factory':frozenset(('ARE ARG AUS AUT BEL BHR BIH BRA CAN CHN CZE DEU DZA EGY ESP FIN '
                              'FRA GBR IDN IND IRN ITA JPN KAZ KOR LBY MEX MYS NLD NZL OMN POL '
                              'PRK QAT RUS SAU SRB SVK SWE TTO TUR UKR USA VEN VNM ZAF').split()),
    'steel-factory':frozenset(('AGO ARE ARG AUS AUT AZE BEL BGD BGR BHR BIH BLR BOL BRA CAN CHE '
                               'CHL CHN CZE DEU DZA EGY ESP FIN FRA GBR GHA GRC GTM HUN IDN IND '
                               'IRN IRQ ITA JPN KAZ KEN KOR KWT LBY LUX MAR MDA MEX MKD MMR MYS '
                               'NGA NLD NOR NZL OMN PAK PER PHL POL PRK PRT QAT ROU RUS SAU SGP '
                               'SRB SVK SVN SWE THA TUR UGA UKR USA UZB VEN VNM ZAF').split()),
}
SEASON_COLORS={'Apr-Sep':'#D55E00','Oct-Mar':'#0072B2'}
MUTED_INK='#666666'
PALE='#F7F7F7'
FIELDS=['stage','hazard','scenario','year','country','season','loss_kg','days_covered']
OUTPUT_FIELDS=['label','Apr-Sep_Abs','Oct-Mar_Abs','Total_Abs','Apr-Sep_Pct','Oct-Mar_Pct',
               'cumulative_start_pct','industry','hazard','scenario','center_year','years_found']


def validate_data(frame):
    """Require the full accepted window; do not infer missing losses as zero."""
    if set(frame.columns)!=set(FIELDS) or frame.empty:
        raise ValueError(f'Expected nonempty input with fields: {FIELDS}')
    if frame.isna().any().any():
        raise ValueError('Missing input fields; missing losses are not zero')
    if set(frame.stage)!=set(STAGES) or set(frame.hazard)!=set(HAZARDS) or set(frame.scenario)!={'ssp370'}:
        raise ValueError('Expected three stages, three hazards and SSP3-7.0')
    if set(frame.year)!=YEARS or set(frame.season)!=set(SEASONS):
        raise ValueError('Expected all years 2055–2065 and both half-year periods')
    if not frame.country.astype(str).str.fullmatch('[A-Z]{3}').all():
        raise ValueError('Expected country alpha-3 identifiers, not aggregate labels')
    if frame.duplicated(['stage','hazard','year','country','season']).any():
        raise ValueError('Duplicate country/year/half-year records')
    if not np.isfinite(frame[['loss_kg','days_covered']]).all().all() or frame.loss_kg.lt(0).any():
        raise ValueError('Loss and coverage must be finite; loss must be nonnegative')
    expected_days=np.array([183 if season=='Apr-Sep' or calendar.isleap(int(year)) else 182
                            for year,season in zip(frame.year,frame.season)])
    if not np.array_equal(frame.days_covered.to_numpy(),expected_days):
        raise ValueError('Incomplete half-year day coverage, including leap days')
    for stage,group in frame.groupby('stage',sort=False):
        countries=sorted(group.country.unique())
        expected=pd.MultiIndex.from_product([HAZARDS,sorted(YEARS),countries,SEASONS])
        actual=pd.MultiIndex.from_frame(group[['hazard','year','country','season']])
        if len(expected.difference(actual)) or len(actual.difference(expected)):
            raise ValueError(f'Incomplete country/hazard/year/half-year coverage for {stage}')


def read_data(path):
    frame=pd.read_csv(path,float_precision='round_trip')
    validate_data(frame)
    for stage,expected in EXPECTED_COUNTRIES.items():
        actual=frozenset(frame.loc[frame.stage.eq(stage),'country'])
        if actual!=expected:
            raise ValueError(f'Incomplete or unexpected country coverage for {stage}: '
                             f'missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    return frame


def cumulative_distribution(annual,industry,hazard):
    """Use validated seasonal totals; retain the adopted accumulation order."""
    panel=annual[annual.stage.eq(industry)&annual.hazard.eq(hazard)]
    if panel.empty:
        return pd.DataFrame(columns=OUTPUT_FIELDS)
    accumulated={}
    years_found=panel.year.nunique()
    for row in panel.sort_values(['year','country','season']).itertuples(index=False):
        key=(row.country,row.season)
        accumulated[key]=accumulated.get(key,0.)+row.loss_kg
    records=[]
    for country in sorted(panel.country.unique()):
        records.append({'label':country,
                        'Apr-Sep_Abs':accumulated[(country,'Apr-Sep')]/years_found,
                        'Oct-Mar_Abs':accumulated[(country,'Oct-Mar')]/years_found})
    country_frame=pd.DataFrame(records)
    country_frame['Total_Abs']=country_frame['Apr-Sep_Abs']+country_frame['Oct-Mar_Abs']
    global_apr_sep=country_frame['Apr-Sep_Abs'].sum()
    global_oct_mar=country_frame['Oct-Mar_Abs'].sum()
    global_total=global_apr_sep+global_oct_mar
    if global_total<=0:
        return pd.DataFrame(columns=OUTPUT_FIELDS)
    # nlargest keeps the earlier alphabetically ordered country if totals tie.
    top_countries=country_frame.nlargest(5,'Total_Abs').copy()
    others={'label':'Other countries',
            'Apr-Sep_Abs':global_apr_sep-top_countries['Apr-Sep_Abs'].sum(),
            'Oct-Mar_Abs':global_oct_mar-top_countries['Oct-Mar_Abs'].sum()}
    global_row={'label':'GLOBAL','Apr-Sep_Abs':global_apr_sep,'Oct-Mar_Abs':global_oct_mar}
    display=pd.concat([pd.DataFrame([global_row]),top_countries,pd.DataFrame([others])],ignore_index=True)
    display['Total_Abs']=display['Apr-Sep_Abs']+display['Oct-Mar_Abs']
    for season in SEASONS:
        display[f'{season}_Pct']=display[f'{season}_Abs']/global_total*100
    cumulative_start=0.
    starts=[0.]
    for _,row in display.iloc[1:].iterrows():
        starts.append(cumulative_start)
        cumulative_start+=row.Total_Abs/global_total*100
    display['cumulative_start_pct']=starts
    display['industry']=industry
    display['hazard']=hazard
    display['scenario']='ssp370'
    display['center_year']=2060
    display['years_found']=years_found
    return display


def compute_plot_values(annual):
    validate_data(annual)
    frames=[cumulative_distribution(annual,stage,hazard) for stage in STAGES for hazard in HAZARDS]
    frames=[frame for frame in frames if not frame.empty]
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_FIELDS)


def style_axis(axis):
    """The accepted axis settings; no changes to the shared Figure 2/3 helper."""
    axis.set_facecolor('white')
    axis.grid(False)
    for side in ['top','right']:
        axis.spines[side].set_visible(False)
    for side in ['left','bottom']:
        axis.spines[side].set_color('#444444')
        axis.spines[side].set_linewidth(.45)
    axis.tick_params(direction='out',length=2.2,width=.45,pad=2)


def render_figure(values):
    apply_figure_style()
    fig,axes=plt.subplots(3,3,figsize=(6.15,4.65),sharex=True)
    hazard_labels={'heatwave':'Heatwave','tropical-cyclone':'Tropical cyclone','wildfire':'Wildfire'}
    industry_labels={'iron-mine':'Iron mines','iron-factory':'Ironmaking facilities','steel-factory':'Steel plants'}
    for i,industry in enumerate(STAGES):
        for j,hazard in enumerate(HAZARDS):
            axis=axes[i,j]
            plotted=values[values.industry.eq(industry)&values.hazard.eq(hazard)]
            if plotted.empty:
                axis.text(.5,.5,'N/A',ha='center',va='center',transform=axis.transAxes,color=MUTED_INK)
                continue
            y_positions=np.arange(len(plotted))
            axis.barh(y_positions,100,left=0,color=PALE,edgecolor='none',height=.64,zorder=1)
            for y,(_,row) in enumerate(plotted.iterrows()):
                start=0. if row.label=='GLOBAL' else row.cumulative_start_pct
                apr_width=row['Apr-Sep_Pct']; oct_width=row['Oct-Mar_Pct']
                axis.barh(y,apr_width,left=start,color=SEASON_COLORS['Apr-Sep'],edgecolor='none',linewidth=0,height=.64,zorder=3)
                axis.barh(y,oct_width,left=start+apr_width,color=SEASON_COLORS['Oct-Mar'],edgecolor='none',linewidth=0,height=.64,zorder=3)
                endpoint=start+apr_width+oct_width
                if row.label not in {'GLOBAL','Other countries'} and endpoint<97:
                    axis.text(endpoint+1.,y,f'{endpoint:.1f}',va='center',ha='left',fontsize=5.5,color=MUTED_INK)
            display_labels=plotted.label.replace({'GLOBAL':'Global','Other countries':'Other'})
            axis.set_yticks(y_positions,display_labels.tolist())
            axis.tick_params(axis='y',labelsize=6.2)
            axis.invert_yaxis()
            axis.set_xlim(0,100)
            axis.set_xticks([0,25,50,75,100])
            style_axis(axis)
            panel_label(axis,i*3+j,x=-.10,y=1.06)
            if i==0:
                axis.set_title(hazard_labels[hazard],pad=4)
            if j==0:
                axis.set_ylabel(industry_labels[industry],labelpad=12,fontweight='bold')
            axis.get_yticklabels()[0].set_fontweight('bold')
    handles=[plt.Rectangle((0,0),1,1,facecolor=SEASON_COLORS[season],edgecolor='none') for season in SEASONS]
    fig.legend(handles,SEASONS,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,handlelength=1.4,columnspacing=1.3)
    fig.supxlabel('Cumulative contribution to global production loss (%)',y=.025)
    fig.subplots_adjust(left=.115,right=.995,top=.90,bottom=.11,wspace=.36,hspace=.28)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-dir',type=Path,default=HERE/'output')
    args=parser.parse_args()
    check_reference_font()
    annual=read_data(args.data)
    values=compute_plot_values(annual)
    figure=render_figure(values)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    apply_figure_style()
    figure.savefig(args.output_dir/'Figure_4.png',dpi=600,bbox_inches='tight',facecolor='white')
    figure.savefig(args.output_dir/'Figure_4.pdf',bbox_inches='tight',facecolor='white')
    plt.close(figure)
    values.to_csv(args.output_dir/'Figure_4_plot_values.csv',index=False)
    print(json.dumps({'output':str(args.output_dir.resolve()),'plot_rows':len(values),'source_rows':len(annual)},indent=2))


if __name__=='__main__':
    main()
