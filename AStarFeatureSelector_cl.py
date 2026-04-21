import numpy as np
import heapq
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows
from LADScorer_cl import lad_score_weighted


class AStarFeatureSelector:
    """
    A* best-first search over binary feature subsets, scored by LAD purity.

    Priority function:  f = g + h_weight * h
        g = len(subset) / n_features   (normalised subset size — penalises large subsets)
        h = 1.0 - lad_score            (heuristic: how far from perfect purity)

    h_weight controls the trade-off between subset size and purity quality.
    Higher values push the search toward maximum purity first.
    Default 5 is a reasonable starting point; tune via ASTAR_H_WEIGHT in config.
    """

    def __init__(self, max_features=None, max_expansions=1110999,
                 h_weight=5, class_weight_disease=1.0):
        self.max_features         = max_features
        self.max_expansions       = max_expansions
        self.h_weight             = h_weight
        self.class_weight_disease = class_weight_disease
        self.X_clean              = None
        self.y_clean              = None

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features

        print(f"\n=== A* LAD Feature Selection Started  "
              f"(h_weight={self.h_weight}  "
              f"class_weight_disease={self.class_weight_disease}) ===")

        pq = []
        heapq.heappush(pq, (0.0, []))
        visited = set()

        best_score             = -1.0
        best_subset_overall    = []
        best_consistent_score  = -1.0
        best_consistent_subset = None
        self.best_subset       = []
        expansions             = 0

        while pq and expansions < self.max_expansions:
            _, subset = heapq.heappop(pq)
            expansions += 1

            key = tuple(sorted(subset))
            if key in visited:
                continue
            visited.add(key)

            score = lad_score_weighted(X, y, subset, self.class_weight_disease)

            if score > best_score:
                best_score          = score
                best_subset_overall = subset[:]
                print(f"[Overall]    New best: {len(subset)} features, "
                      f"score={best_score:.4f}, "
                      f"features: {[bin_feature_names[i] for i in subset]}")

            if subset and check_consistency(X, y, subset):
                if score > best_consistent_score:
                    best_consistent_score  = score
                    best_consistent_subset = subset[:]
                    print(f"[Consistent] New best discriminating: "
                          f"{len(subset)} features, "
                          f"score={best_consistent_score:.4f}, "
                          f"features: {[bin_feature_names[i] for i in subset]}")

            if len(subset) >= self.max_features:
                continue

            for feat in range(n_features):
                if feat in subset:
                    continue
                new_subset = subset + [feat]
                g     = len(new_subset) / n_features
                h     = 1.0 - score
                f_new = g + self.h_weight * h
                heapq.heappush(pq, (f_new, new_subset))

        if best_consistent_subset is not None:
            self.best_subset = best_consistent_subset
            self.X_clean     = X
            self.y_clean     = y
            print(f"\n[Result] Using best discriminating subset: "
                  f"{len(self.best_subset)} features, "
                  f"score={best_consistent_score:.4f}")
        else:
            self.best_subset = best_subset_overall
            print(f"\n[ConsistencyWarning] No fully discriminating subset found "
                  f"after {expansions} expansions. Falling back to best overall "
                  f"({len(self.best_subset)} features, score={best_score:.4f}) "
                  f"and removing conflicting rows.")
            self.X_clean, self.y_clean, _ = remove_conflicting_rows(
                X, y, self.best_subset, verbose=True
            )

        print(f"A* finished after {expansions} expansions.\n")
        return self

    def transform(self, X):
        return X[:, self.best_subset] if hasattr(self, "best_subset") else X