# Formative 3: Probability Distributions, Bayesian Probability & Gradient Descent

Group submission for the Machine Learning formative (ALU).

## Repository structure

```
├── data/
│   ├── GaltonFamilies.csv          # Part 1: Galton parents/children heights
│   └── IMDB Dataset.csv            # Part 2: 50k IMDb reviews 
├── src/                            # Modular, DRY implementations
│   ├── em_gmm.py                   # EM algorithm from scratch (Part 1)
│   ├── bayes_sentiment.py          # Bayes' theorem, basic Python only (Part 2)
│   └── gradient_descent.py         # Matrix-form gradient descent (Parts 3 & 4)
├── notebooks/
│   ├── part1_em_gaussian_mixture.ipynb
│   ├── part2_bayesian_sentiment.ipynb
│   └── part4_gradient_descent.ipynb                
└── README.md
```

Setup: `pip install numpy pandas scipy matplotlib`, then run the notebooks
from the `notebooks/` folder. `IMDB Dataset.csv` (66 MB) is git-ignored, so
download it from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
and put it in `data/`.

---

## Part 1: EM algorithm on a Gaussian mixture (heights)

We pooled fathers' and children's (daughters') heights from the Galton
dataset into one unlabeled sample and fit a 2-component Gaussian mixture
with an EM algorithm we wrote from scratch (`src/em_gmm.py`).

*Why daughters?* Galton's "children" are adult offspring, and adult sons are
about as tall as fathers (both around 69 in). So if we mixed all children
with fathers, the true bimodality in the data would be gender, not
generation. The notebook actually demonstrates this at the end. Controlling
for gender gives two well defined populations: children at about 64.1 in and
fathers at about 69.3 in.

**Should you just split at the global mean and average the two piles? No.**

- The two populations overlap. Tall children land above the line and short
  fathers below it, so both piles get contaminated (misclassification).
- Averaging a cut pile means averaging a *truncated* Gaussian, which is a
  biased estimate. It pushes the two estimated means artificially apart.
- A hard threshold ignores that the groups have different variances and
  different sizes (mixing weights pi1, pi2).
- EM makes soft, probabilistic assignments instead: every point gets a
  responsibility (the posterior probability of belonging to each component)
  and contributes fractionally to both parameter updates.

**How our EM works**

1. **Initialization:** means/variances from a crude split, pi1 = pi2 = 0.5.
2. **E-step:** compute responsibilities with Bayes' theorem:
   gamma_ik = pi_k N(x_i | mu_k, sigma_k^2) / sum_j pi_j N(x_i | mu_j, sigma_j^2).
3. **M-step:** update mu_k, sigma_k^2 and pi_k with responsibility-weighted
   (soft count) averages.
4. **Stopping:** iterate until the log-likelihood, which never decreases,
   changes by less than 1e-6.

The notebook prints the required tracking table (iterations 0, 1, 2 and the
final state with mu1, mu2, sigma1^2, sigma2^2, pi1, pi2 and log-likelihood)
and has a `report(height)` cell that takes any test height the coach gives
us live and prints the exact posterior probability of each population.

## Part 2: Bayesian probability on IMDb reviews

- Positive keywords: **excellent, amazing, wonderful**. Negative: **awful,
  terrible, waste**. We went for the superlatives people use when they loved
  a film vs the vocabulary of angry reviews ("waste of time/money").
- We compute **P(Positive | keyword)** only.
- The implementation is pure basic Python (`csv`, `re`, arithmetic). One
  pass over the 50,000 reviews counts, for each keyword, how many reviews
  contain it and how many of those are positive. Then
  `posterior = likelihood * prior / marginal`.
- The notebook prints the full table (prior P(Positive), likelihood
  P(kw|Positive), marginal P(kw), posterior P(Positive|kw)) for every
  keyword, plus a sanity check that Bayes' theorem matches direct counting.

## Part 3: Manual gradient descent (matrix form)

`docs/part3_manual_gradient_descent.md` shows, with no skipped arithmetic:
the chain rule derivation of dJ/dm = (2/n) X^T e and dJ/db = (2/n) e, and
four full iterations (one per group member) with every intermediate vector
(predictions, error, cost, both gradients, updates) at learning rate 0.01.
The MSE falls 61 -> 6.50 -> 2.50 -> 2.16 -> 2.10.

## Part 4: Gradient descent in code

`notebooks/part4_gradient_descent.ipynb` reproduces Part 3 exactly in Python:

- SciPy (`scipy.differentiate.derivative`) numerically differentiates the
  MSE cost function and confirms our analytic matrix gradients.
- The update loop prints every intermediate step, nothing is over-abstracted.
- Matplotlib figures: how m and b change over the iterations, and how the error changes over the iterations.

## Contributions

| Member | Contribution |
|--------|--------------|
| Member 1 | Part 1 EM implementation + iteration 1 of Part 3 |
| Member 2 | Part 2 Bayes implementation + iteration 2 of Part 3 |
| Member 3 | Part 4 SciPy/plots + iteration 3 of Part 3 |
| Member 4 | README/presentation + iteration 4 of Part 3 |

