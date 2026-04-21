import numpy as np
import heapq
from sklearn.feature_selection import mutual_info_classif
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows
from LADScorer_cl import lad_score_weighted


class MutualInfoAStarSelector:
    """
    Two-phase selector: MI pre-ranking + LAD-scored A* search.

    Phase 1: Rank all binary features by Mutual Information.
    Phase 2: A* over the full pool, expanding in MI-ranked order.
             A small rank_penalty breaks ties in favour of higher-MI features.

    Priority:  f = g + h_weight * h + rank_penalty
        g            = len(subset) / n_features
        h            = 1.0 - lad_score
        rank_penalty = (mi_rank / n_features) * 0.1
    """

    def __init__(self, max_expansions=10000, h_weight=5,
                 class_weight_disease=1.0): #1110999
        self.max_expansions       = max_expansions
        self.h_weight             = h_weight
        self.class_weight_disease = class_weight_disease
        self.X_clean              = None
        self.y_clean              = None

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n_features = X.shape[1]

        print(f"\n=== MutualInfo A* LAD Feature Selection Started  "
              f"(h_weight={self.h_weight}  "
              f"class_weight_disease={self.class_weight_disease}) ===")
        print("Phase 1: Computing Mutual Information rankings...")

        mi     = mutual_info_classif(X, y, discrete_features=True)
        ranked = np.argsort(mi)[::-1]

        print(f"Top 10 features by MI: {ranked[:10].tolist()}")
        print(f"Candidate pool: all {n_features} features, "
              f"expanding in MI-ranked order\n")
        print("Phase 2: Running A* search...")

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

            if len(subset) >= n_features:
                continue

            for rank, feat in enumerate(ranked):
                if feat in subset:
                    continue
                new_subset   = subset + [feat]
                g            = len(new_subset) / n_features
                h            = 1.0 - score
                rank_penalty = rank / n_features * 0.1
                f_new        = g + self.h_weight * h + rank_penalty
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

        print(f"MutualInfo A* finished after {expansions} expansions.\n")
        return self

    def transform(self, X):
        return X[:, self.best_subset] if self.best_subset else X

    def fit_transform(self, X, y, bin_feature_names):
        self.fit(X, y, bin_feature_names)
        return self.transform(X)