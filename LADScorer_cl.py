"""
LADScorer — shared purity-based subset scorer used by all four selectors.

Two scoring functions are provided:

lad_score(X, y, subset)
    Symmetric coverage-weighted average purity.
    Treats both classes equally.
    Range [0.5, 1.0].

lad_score_weighted(X, y, subset, class_weight_disease=1.0)
    Same as lad_score but patterns whose majority label is disease (1)
    have their weight multiplied by class_weight_disease before
    contributing to the total.

    class_weight_disease=1.0  -> identical to lad_score (symmetric)
    class_weight_disease=1.5  -> disease patterns count 50% more
    class_weight_disease=2.0  -> disease patterns count double

    The score is renormalised after weighting so it stays in [0, 1].
    This biases feature selection toward subsets that better discriminate
    disease, directly reducing false negatives at the cost of some
    increase in false positives.

All selectors accept an optional class_weight_disease parameter and
call lad_score_weighted internally. Setting it to 1.0 (default) gives
identical behaviour to the original symmetric scorer.
"""

import numpy as np


def lad_score(X, y, subset):
    """
    Symmetric coverage-weighted average purity.
    Equivalent to lad_score_weighted with class_weight_disease=1.0.
    Kept for backwards compatibility with any code that imports it directly.
    """
    return lad_score_weighted(X, y, subset, class_weight_disease=1.0)


def lad_score_weighted(X, y, subset, class_weight_disease=1.0):
    """
    Coverage-weighted average purity with optional disease-class boost.

    For each unique binary pattern in X[:, subset]:
      1. Compute purity  = fraction of rows with majority label
      2. Compute weight  = (rows covered / total rows)
                          * class_weight_disease  if majority label == 1
                          * 1.0                   if majority label == 0
      3. Accumulate purity * weight

    Normalise by total weight sum so the score stays in [0, 1].

    Parameters
    ----------
    X                    : (n_samples, n_features) binarized matrix
    y                    : (n_samples,) binary labels  0=healthy, 1=disease
    subset               : list[int]  column indices to evaluate
    class_weight_disease : float >= 1.0  multiplier on disease-pattern weight

    Returns
    -------
    float in [0, 1]
    """
    if not subset:
        return 0.0

    n        = len(y)
    Xsub     = X[:, subset]
    total    = 0.0
    w_sum    = 0.0

    for pattern in np.unique(Xsub, axis=0):
        mask    = np.all(Xsub == pattern, axis=1)
        covered = y[mask]
        n_cov   = len(covered)
        if n_cov == 0:
            continue

        labels, counts = np.unique(covered, return_counts=True)
        majority_label = labels[np.argmax(counts)]
        purity         = counts.max() / n_cov

        # Base weight: fraction of training rows this pattern covers
        base_w = n_cov / n

        # Boost if majority label is disease
        boost  = class_weight_disease if int(majority_label) == 1 else 1.0
        w      = base_w * boost

        total += purity * w
        w_sum += w

    if w_sum == 0.0:
        return 0.0

    # Normalise so score stays in [0, 1]
    return float(total / w_sum)