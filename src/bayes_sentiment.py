"""
Part 2: Bayes' theorem on the IMDb 50k movie reviews dataset.

We compute P(Positive | keyword) for each keyword we picked, using only
basic Python (csv module, strings, arithmetic). No ML libraries anywhere.

    P(Positive | kw) = P(kw | Positive) * P(Positive) / P(kw)
"""

import csv
import re

POSITIVE_KEYWORDS = ["excellent", "amazing", "wonderful"]
NEGATIVE_KEYWORDS = ["awful", "terrible", "waste"]


def load_reviews(path):
    """Go through the CSV and yield (review_text_lowercase, sentiment)."""
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                       # skip the header row
        for row in reader:
            yield row[0].lower(), row[1]   # (review, 'positive'/'negative')


def contains_word(text, word):
    """Whole word match, so 'waste' doesn't count inside 'wasteland'."""
    return re.search(r"\b" + word + r"\b", text) is not None


def count_statistics(path, keywords):
    """
    One pass over the whole dataset. We count:
      - total reviews and how many are positive
      - for each keyword: how many reviews contain it, and how many of
        those are positive
    """
    n_total = 0
    n_positive = 0
    kw_count = {kw: 0 for kw in keywords}
    kw_and_pos = {kw: 0 for kw in keywords}

    for text, sentiment in load_reviews(path):
        n_total += 1
        is_pos = sentiment == "positive"
        if is_pos:
            n_positive += 1
        for kw in keywords:
            if contains_word(text, kw):
                kw_count[kw] += 1
                if is_pos:
                    kw_and_pos[kw] += 1
    return n_total, n_positive, kw_count, kw_and_pos


def bayes_table(path, keywords):
    """
    Compute the four required probabilities for each keyword:
      prior      P(Positive)       = n_positive / n_total
      likelihood P(kw | Positive)  = n_kw_and_pos / n_positive
      marginal   P(kw)             = n_kw / n_total
      posterior  P(Positive | kw)  = likelihood * prior / marginal
    Returns a list of dicts, one per keyword.
    """
    n_total, n_positive, kw_count, kw_and_pos = count_statistics(path, keywords)
    prior = n_positive / n_total
    rows = []
    for kw in keywords:
        likelihood = kw_and_pos[kw] / n_positive
        marginal = kw_count[kw] / n_total
        posterior = likelihood * prior / marginal
        rows.append(dict(keyword=kw, prior=prior, likelihood=likelihood,
                         marginal=marginal, posterior=posterior,
                         n_reviews_with_kw=kw_count[kw]))
    return rows


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/IMDB Dataset.csv"
    for row in bayes_table(path, POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS):
        print(f"{row['keyword']:>10}: prior={row['prior']:.4f} "
              f"likelihood={row['likelihood']:.4f} "
              f"marginal={row['marginal']:.4f} "
              f"posterior P(Pos|kw)={row['posterior']:.4f}")
