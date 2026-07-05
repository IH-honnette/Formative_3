# %% [markdown]
# # Part 2: Bayesian Probability on IMDb Movie Reviews
#
# **Dataset:** [IMDb 50k movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
#
# **Our group decisions**
# * Positive keywords: `excellent`, `amazing`, `wonderful`. These are the
#   superlatives people reach for when they loved a film.
# * Negative keywords: `awful`, `terrible`, `waste`. We picked "waste"
#   because in reviews it almost always shows up as "waste of time/money".
# * We compute **P(Positive | keyword)** only, not the negative direction.
#
# Bayes' theorem:
#
# $$P(\text{Positive} \mid kw) = \frac{P(kw \mid \text{Positive}) \; P(\text{Positive})}{P(kw)}$$
#
# The implementation uses **basic Python only** (`csv`, `re`, arithmetic),
# see `src/bayes_sentiment.py`. We match keywords as whole words on
# lowercased text so that e.g. "waste" does not fire inside "wasteland".

# %%
import sys
sys.path.append("../src")

from bayes_sentiment import (POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS,
                             count_statistics, bayes_table)

DATA = "../data/IMDB Dataset.csv"
KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS
print("positive keywords:", POSITIVE_KEYWORDS)
print("negative keywords:", NEGATIVE_KEYWORDS)

# %% [markdown]
# ## 1. Raw counts (single pass over 50,000 reviews)

# %%
n_total, n_positive, kw_count, kw_and_pos = count_statistics(DATA, KEYWORDS)
print(f"total reviews          n            = {n_total}")
print(f"positive reviews       n_pos        = {n_positive}")
print(f"prior P(Positive)                   = {n_positive/n_total:.4f}\n")
for kw in KEYWORDS:
    print(f"{kw:>10}: appears in {kw_count[kw]:5d} reviews, "
          f"{kw_and_pos[kw]:5d} of them positive")

# %% [markdown]
# ## 2. Prior, likelihood, marginal and posterior for every keyword

# %%
rows = bayes_table(DATA, KEYWORDS)

header = f"{'keyword':>10} | {'P(Pos)':>7} | {'P(kw|Pos)':>9} | {'P(kw)':>7} | {'P(Pos|kw)':>9}"
print(header); print("-" * len(header))
for r in rows:
    print(f"{r['keyword']:>10} | {r['prior']:7.4f} | {r['likelihood']:9.4f} "
          f"| {r['marginal']:7.4f} | {r['posterior']:9.4f}")

# %% [markdown]
# ## 3. Sanity check: Bayes vs direct counting
#
# $P(\text{Pos}\mid kw)$ can also be computed directly as
# (positive reviews containing kw) / (reviews containing kw). Both routes
# have to agree, so this is a nice check that our Bayes implementation is
# actually correct.

# %%
for kw in KEYWORDS:
    direct = kw_and_pos[kw] / kw_count[kw]
    bayes = next(r["posterior"] for r in rows if r["keyword"] == kw)
    assert abs(direct - bayes) < 1e-12, kw
    print(f"{kw:>10}: direct={direct:.6f}  bayes={bayes:.6f}  match")
print("\nAll posteriors verified.")

# %% [markdown]
# ### Interpretation
# * The prior is about 0.5 (the dataset is balanced by construction), so any
#   posterior far from 0.5 is signal carried by the keyword itself.
# * The positive keywords push the posterior well **above** the prior
#   (a review containing "excellent" is far more likely to be positive),
#   and the negative keywords push it well **below**.
# * This is the same evidence-updating logic that Naive Bayes classifiers
#   use, we just did it for one keyword at a time with nothing but counting.
