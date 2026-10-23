"""Statistical inference for flight-level descriptors clustered by collection day."""
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def mixed_model(df, y, x, covariates, group='day'):
    """Random-intercept model y ~ x + covariates; returns the standardized coefficient of x, its CI and p value."""
    d = df.dropna(subset=[y, x] + covariates).copy()
    d['_y'] = (d[y] - d[y].mean()) / d[y].std(); d['_x'] = (d[x] - d[x].mean()) / d[x].std()
    formula = '_y ~ _x' + ''.join(f' + {c}' for c in covariates)
    for method in (None, 'powell', 'nm'):
        fit = smf.mixedlm(formula, d, groups=d[group]).fit(**({} if method is None else {'method': method}))
        if fit.converged:
            break
    lo, hi = fit.conf_int().loc['_x']
    return fit.params['_x'], lo, hi, fit.pvalues['_x']


def within_between(df, y, x, covariates, group='day'):
    """Mundlak decomposition of x into a group mean and a within-group deviation."""
    d = df.copy()
    d['x_between'] = d.groupby(group)[x].transform('mean'); d['x_within'] = d[x] - d['x_between']
    formula = f'{y} ~ x_within + x_between' + ''.join(f' + {c}' for c in covariates)
    return smf.mixedlm(formula, d, groups=d[group]).fit()


def leave_one_group_out(df, y, x, covariates, group='day'):
    return [mixed_model(df[df[group] != g], y, x, covariates, group) for g in df[group].unique()]


def fdr(pvalues):
    return multipletests(pvalues, method='fdr_bh')[1]


def paired_difference(df, y_mech, y_ref, x, covariates, group='day'):
    """Day fixed-effects model of the within-flight standardized channel difference, day-clustered errors."""
    d = df.copy(); z = lambda s: (s - s.mean()) / s.std()
    d['_diff'] = z(d[y_mech]) - z(d[y_ref])
    fit = smf.ols('_diff ~ ' + x + ''.join(f' + {c}' for c in covariates) + f' + C({group})', d).fit(
        cov_type='cluster', cov_kwds={'groups': d[group]}, use_t=True)
    return fit.params[x] * d[x].std(), fit.conf_int().loc[x].values * d[x].std(), fit.pvalues[x]


def propagate_estimation_error(df, y, se, x, covariates, n_draws=300, seed=0, group='day'):
    """Monte Carlo propagation of per-flight exponent standard errors into the mixed-model coefficient."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_draws):
        d = df.copy(); d[y] = d[y] + rng.normal(0, d[se])
        out.append(mixed_model(d, y, x, covariates, group))
    return np.array(out)
