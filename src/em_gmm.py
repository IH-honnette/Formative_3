"""
EM algorithm for a mixture of two Gaussians in 1D, written from scratch.
Only numpy is used, no sklearn.

This is for Part 1. We take the Galton height data (fathers + children),
pretend we lost the labels, and let EM figure out the two groups on its own.
"""

import numpy as np


def gaussian_pdf(x, mu, var):
    """Normal distribution density N(x | mu, var)."""
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def log_likelihood(x, mu1, mu2, var1, var2, pi1, pi2):
    """Log-likelihood of the whole dataset under the current mixture."""
    mix = pi1 * gaussian_pdf(x, mu1, var1) + pi2 * gaussian_pdf(x, mu2, var2)
    return np.sum(np.log(mix))


def e_step(x, mu1, mu2, var1, var2, pi1, pi2):
    """
    E-step: for every point, compute the probability that it belongs to
    component 1 or component 2 (the "responsibilities"). This is just
    Bayes' theorem:

        gamma_1(x) = pi1 * N(x|mu1,var1) / (pi1*N(x|mu1,var1) + pi2*N(x|mu2,var2))
    """
    w1 = pi1 * gaussian_pdf(x, mu1, var1)
    w2 = pi2 * gaussian_pdf(x, mu2, var2)
    total = w1 + w2
    return w1 / total, w2 / total


def m_step(x, gamma1, gamma2):
    """
    M-step: re-estimate the parameters, but instead of normal counts we use
    the responsibilities as fractional (soft) counts. So a point that is 70%
    component 1 contributes 0.7 of itself to component 1's mean.
    """
    n1, n2 = gamma1.sum(), gamma2.sum()
    mu1 = (gamma1 * x).sum() / n1
    mu2 = (gamma2 * x).sum() / n2
    var1 = (gamma1 * (x - mu1) ** 2).sum() / n1
    var2 = (gamma2 * (x - mu2) ** 2).sum() / n2
    pi1, pi2 = n1 / len(x), n2 / len(x)
    return mu1, mu2, var1, var2, pi1, pi2


def initialize(x, seed=42):
    """
    Starting point for EM. We just split the data at the global mean and use
    each half's mean/variance. It's a rough start on purpose, EM is supposed
    to fix it. Mixing weights start at 50/50.
    """
    rng = np.random.default_rng(seed)
    split = x.mean()
    lo, hi = x[x < split], x[x >= split]
    mu1, mu2 = lo.mean(), hi.mean()
    var1, var2 = lo.var(), hi.var()
    pi1 = pi2 = 0.5
    return mu1, mu2, var1, var2, pi1, pi2


def fit_em(x, max_iter=200, tol=1e-6, seed=42, verbose=False):
    """
    Run EM until the log-likelihood basically stops improving (change below
    tol), or until max_iter.

    Returns (params, history). history is a list of dicts, one per iteration
    (iteration 0 is the initialization), with mu1, mu2, var1, var2, pi1, pi2
    and the log-likelihood. We use it to print the tracking table.
    """
    mu1, mu2, var1, var2, pi1, pi2 = initialize(x, seed)
    history = []

    def record(it):
        history.append(dict(iteration=it, mu1=mu1, mu2=mu2, var1=var1,
                            var2=var2, pi1=pi1, pi2=pi2,
                            log_likelihood=log_likelihood(x, mu1, mu2, var1,
                                                          var2, pi1, pi2)))

    record(0)
    for it in range(1, max_iter + 1):
        gamma1, gamma2 = e_step(x, mu1, mu2, var1, var2, pi1, pi2)
        mu1, mu2, var1, var2, pi1, pi2 = m_step(x, gamma1, gamma2)
        record(it)
        delta = history[-1]["log_likelihood"] - history[-2]["log_likelihood"]
        if verbose:
            print(f"iter {it:3d}  logL={history[-1]['log_likelihood']:.4f}  "
                  f"delta={delta:.2e}")
        if abs(delta) < tol:
            break

    params = dict(mu1=mu1, mu2=mu2, var1=var1, var2=var2, pi1=pi1, pi2=pi2)
    return params, history


def classify_height(height, params, labels=("Children", "Fathers")):
    """
    Take one test height and return the posterior probability that it came
    from each component (Bayes' theorem again). Returns {label: probability}.
    """
    w1 = params["pi1"] * gaussian_pdf(height, params["mu1"], params["var1"])
    w2 = params["pi2"] * gaussian_pdf(height, params["mu2"], params["var2"])
    total = w1 + w2
    return {labels[0]: w1 / total, labels[1]: w2 / total}


def naive_mean_split(x):
    """
    The naive method we argue against in the presentation: draw a line at
    the global mean, split into two piles, take each pile's mean.
    """
    split = x.mean()
    return x[x < split].mean(), x[x >= split].mean(), split
