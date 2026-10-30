"""Original Figure 6 display inference and OLS band, with no project imports.

Country resampling is conditional on this simulated scenario; it does not model
trade-network dependence, ABM uncertainty or causality.
"""
import numpy as np
from scipy import stats

SEED=20260913
BOOTSTRAPS=20_000
PERMUTATIONS=99_999


def bootstrap_ci(x,y,*,rank=False,n_resamples=BOOTSTRAPS,seed=SEED):
    """Original paired-country percentile bootstrap; recompute tied ranks."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    rng=np.random.default_rng(seed)
    estimates=[]
    for start in range(0,n_resamples,500):
        indices=rng.integers(0,len(x),size=(min(500,n_resamples-start),len(x)))
        bx,by=x[indices],y[indices]
        if rank:
            bx,by=stats.rankdata(bx,axis=1),stats.rankdata(by,axis=1)
        bw=np.ones_like(bx)
        bw/=bw.sum(axis=1,keepdims=True)
        bx=bx-np.sum(bw*bx,axis=1,keepdims=True)
        by=by-np.sum(bw*by,axis=1,keepdims=True)
        denominator=np.sqrt(np.sum(bw*bx**2,axis=1)*np.sum(bw*by**2,axis=1))
        if (denominator<=0).any():
            raise ValueError('Undefined correlation in bootstrap sample')
        estimates.extend(np.sum(bw*bx*by,axis=1)/denominator)
    return tuple(float(v) for v in np.quantile(estimates,[.025,.975]))


def permutation_spearman(x,y,*,n_resamples=PERMUTATIONS,seed=SEED+1):
    """Original two-sided pairing test, with ties and plus-one correction."""
    rx,ry=stats.rankdata(x),stats.rankdata(y)
    rx,ry=rx-rx.mean(),ry-ry.mean()
    denominator=np.linalg.norm(rx)*np.linalg.norm(ry)
    if denominator==0:
        raise ValueError('Spearman correlation requires nonconstant variables')
    estimate=float(rx@ry/denominator)
    rng=np.random.default_rng(seed)
    extreme=sum(abs(float(rx@rng.permutation(ry)/denominator))>=abs(estimate)-1e-12 for _ in range(n_resamples))
    return {'estimate':estimate,'p_value':(extreme+1)/(n_resamples+1),
            'extreme_count':int(extreme),'permutations':n_resamples}


def calculate_statistics(scores):
    """Recalculate the two displayed correlations in deterministic ISO order."""
    data=scores.sort_values('Country')
    x,y=data.E2ECI.to_numpy(float),data.satisfaction_pct.to_numpy(float)
    if len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x)==0 or np.ptp(y)==0:
        raise ValueError('Correlation requires complete nonconstant paired data')
    fit=stats.pearsonr(x,y)
    plow,phigh=bootstrap_ci(x,y)
    spearman=permutation_spearman(x,y)
    slow,shigh=bootstrap_ci(x,y,rank=True,seed=SEED+1)
    return {'n':len(x),'resampling':{'seed':SEED,'bootstrap_samples':BOOTSTRAPS,
            'permutations':PERMUTATIONS,'endpoint_tolerance_percentage_points':1e-8,
            'unit':'country; deterministic ISO order; Spearman seed offset +1'},
            'results':{'pearson':{'estimate':float(fit.statistic),'p_value':float(fit.pvalue),
                                  'ci_low':plow,'ci_high':phigh},
                       'spearman':{**spearman,'ci_low':slow,'ci_high':shigh}},
            'interpretation_boundary':'Conditional cross-country association, not causality or model/network uncertainty.'}


def ols_band(x,y,grid):
    """Original 95% OLS mean-fit band, not a prediction or correlation band."""
    fit=stats.linregress(x,y)
    fitted=fit.intercept+fit.slope*x
    residual_se=np.sqrt(np.sum((y-fitted)**2)/(len(x)-2))
    centered_ss=np.sum((x-x.mean())**2)
    grid_fit=fit.intercept+fit.slope*grid
    mean_se=residual_se*np.sqrt(1./len(x)+(grid-x.mean())**2/centered_ss)
    margin=stats.t.ppf(.975,len(x)-2)*mean_se
    return grid_fit,grid_fit-margin,grid_fit+margin,fit
