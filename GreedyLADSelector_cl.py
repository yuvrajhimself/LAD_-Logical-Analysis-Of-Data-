import numpy as np
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows
from LADScorer_cl import lad_score_weighted


class GreedyLADSelector:
    """
    Forward stepwise feature selector scored by LAD purity.

    class_weight_disease > 1.0 boosts the score contribution of patterns
    whose majority label is disease (1), biasing selection toward features
    that better discriminate sick patients — directly reducing false negatives.
    """

    def __init__(self, class_weight_disease=1.0):
        self.class_weight_disease = class_weight_disease
        self.best_subset          = []
        self.X_clean              = None
        self.y_clean              = None

    def _score(self, X, y, subset):
        return lad_score_weighted(X, y, subset, self.class_weight_disease)

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n         = X.shape[1]
        remaining = list(range(n))
        self.best_subset       = []
        best_overall           = 0.0
        best_consistent_score  = -1.0
        best_consistent_subset = None

        print(f"\n=== Greedy LAD Feature Selection Started "
              f"(class_weight_disease={self.class_weight_disease}) ===")

        while remaining:
            best_f     = None
            best_score = best_overall

            for f in remaining:
                subset = self.best_subset + [f]
                score  = self._score(X, y, subset)
                if score > best_score:
                    best_score = score
                    best_f     = f

            if best_f is None:
                print("No improvement. Stopping.")
                break

            self.best_subset.append(best_f)
            remaining.remove(best_f)
            best_overall = best_score
            print(f"[Overall]    Added: {bin_feature_names[best_f]} "
                  f"-> LAD score = {best_score:.4f}")

            if check_consistency(X, y, self.best_subset):
                if best_score > best_consistent_score:
                    best_consistent_score  = best_score
                    best_consistent_subset = self.best_subset[:]
                    print(f"[Consistent] New best discriminating: "
                          f"{len(self.best_subset)} feature(s), "
                          f"score={best_consistent_score:.4f}")

        if best_consistent_subset is not None:
            self.best_subset = best_consistent_subset
            self.X_clean     = X
            self.y_clean     = y
            print(f"\n[Result] Using best discriminating subset: "
                  f"{len(self.best_subset)} features, "
                  f"score={best_consistent_score:.4f}")
        else:
            print(f"\n[ConsistencyWarning] No fully discriminating subset found. "
                  f"Falling back to best overall subset "
                  f"({len(self.best_subset)} features, score={best_overall:.4f}) "
                  f"and removing conflicting rows.")
            self.X_clean, self.y_clean, _ = remove_conflicting_rows(
                X, y, self.best_subset, verbose=True
            )

        print(f"Final selected {len(self.best_subset)} features.\n")
        return self

    def transform(self, X):
        return X[:, self.best_subset]

    def fit_transform(self, X, y, bin_feature_names):
        self.fit(X, y, bin_feature_names)
        return self.transform(X)