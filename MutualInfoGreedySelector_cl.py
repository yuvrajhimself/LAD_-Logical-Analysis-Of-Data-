import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows
from LADScorer_cl import lad_score_weighted


class MutualInfoGreedySelector:
    """
    Two-phase selector: MI pre-ranking + LAD-scored greedy walk.

    Phase 1: Rank features by Mutual Information with the target.
    Phase 2: Walk ranked list in order; keep each feature if lad_score
             on the accumulated subset meets or improves the current best.
    """

    def __init__(self, class_weight_disease=1.0):
        self.class_weight_disease = class_weight_disease
        self.X_clean              = None
        self.y_clean              = None

    def fit(self, X, y):
        mi     = mutual_info_classif(X, y, discrete_features=True)
        ranked = np.argsort(mi)[::-1]

        self.best_subset       = []
        best_overall           = 0.0
        best_consistent_score  = -1.0
        best_consistent_subset = None

        print(f"\n=== MutualInfo Greedy LAD Feature Selection Started "
              f"(class_weight_disease={self.class_weight_disease}) ===")
        print(f"Feature ranking by MI (best first): {ranked.tolist()}\n")

        for f in ranked:
            trial = self.best_subset + [f]
            score = lad_score_weighted(X, y, trial, self.class_weight_disease)

            if score >= best_overall:
                self.best_subset.append(f)
                best_overall = score
                print(f"[Overall]    Added feature {f} "
                      f"-> LAD score = {best_overall:.4f}")

                if check_consistency(X, y, self.best_subset):
                    if score > best_consistent_score:
                        best_consistent_score  = score
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

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)