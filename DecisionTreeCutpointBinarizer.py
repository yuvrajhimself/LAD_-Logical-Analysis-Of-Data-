import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

class DecisionTreeCutpointBinarizer:
    def __init__(self,
                 mode="two",  # now supports "one", "two", "all"
                 max_depth=5,
                 min_samples_leaf=15,
                 min_impurity_decrease=0.005,
                 min_support=15,
                 edge_fraction=0.05,
                 min_interval_fraction=0.12,
                 random_state=42):

        assert mode in ("one", "two", "all"), "mode must be 'one', 'two', or 'all'"
        self.mode = mode
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_support = min_support
        self.edge_fraction = edge_fraction
        self.min_interval_fraction = min_interval_fraction
        self.random_state = random_state

        self.cutpoints = {}  # cut_id → (feat_idx, threshold)
        self.feature_names = None

    def get_cutpoints(self):
        return self.cutpoints

    def print_cutpoints_readable(self):
        print("\n=== DISCOVERED CUTPOINTS (Binarization Stage) ===")
        if len(self.cutpoints) == 0:
            print("No cutpoints found!")
            return
        for cut_id, (feat_idx, thresh) in self.cutpoints.items():
            feat_name = self.feature_names[feat_idx] if self.feature_names is not None else f"feature_{feat_idx}"
            print(f"Cut {cut_id:3d}:  {feat_name} <= {thresh:.6f}")
        print(f"Total binary features created: {len(self.cutpoints)}\n")

    def _tree_thresholds(self, x, y):
        x = x.reshape(-1, 1)
        sample_weight = compute_sample_weight("balanced", y)

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            class_weight="balanced"
        )
        tree.fit(x, y, sample_weight=sample_weight)

        thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
        thresholds = np.unique(np.round(thresholds, 8))

        if len(thresholds) == 0:
            return np.array([]), np.array([])

        gains = []
        for t in thresholds:
            left_mask = x.flatten() <= t
            n_left = left_mask.sum()
            n_right = len(x) - n_left
            if n_left < 2 or n_right < 2:
                gains.append(0)
                continue

            p_left = y[left_mask].mean()
            p_right = y[~left_mask].mean()
            p_total = y.mean()

            gain = (n_left + n_right) * (
                p_total * (1 - p_total) -
                (n_left / (n_left + n_right)) * p_left * (1 - p_left) -
                (n_right / (n_left + n_right)) * p_right * (1 - p_right)
            )
            gains.append(gain)

        order = np.argsort(thresholds)
        return thresholds[order], np.array(gains)[order]
    
    def _prune_thresholds(self, x, thresholds, gains):
        if len(thresholds) == 0:
            return [], []

        pruned_thresh = []
        pruned_gains = []

        xmin, xmax = np.min(x), np.max(x)
        range_x = max(xmax - xmin, 1e-8)
        edge_min = xmin + self.edge_fraction * range_x
        edge_max = xmax - self.edge_fraction * range_x

        for t, g in zip(thresholds, gains):
            if t <= edge_min or t >= edge_max:
                continue
            left_count = np.sum(x <= t)
            right_count = len(x) - left_count
            if left_count < self.min_support or right_count < self.min_support:
                continue
            pruned_thresh.append(t)
            pruned_gains.append(g)

        return pruned_thresh, pruned_gains

    def _select_cutpoints(self, pruned_thresh, pruned_gains, x):
        if len(pruned_thresh) == 0:
            return []

        # Sort by threshold value
        sorted_idx = np.argsort(pruned_thresh)
        thresh_sorted = np.array(pruned_thresh)[sorted_idx]
        gains_sorted = np.array(pruned_gains)[sorted_idx]

        if self.mode == "all":
            return thresh_sorted.tolist()  # Return ALL pruned thresholds

        if self.mode == "one" or len(thresh_sorted) <= 1:
            # Pick the one with highest gain
            best_idx = np.argmax(gains_sorted)
            return [thresh_sorted[best_idx]]

        # Mode "two": keep lower and upper, but only if sufficiently apart
        lower = thresh_sorted[0]
        upper = thresh_sorted[-1]
        range_x = np.ptp(x)

        if len(thresh_sorted) == 1 or (upper - lower) < self.min_interval_fraction * range_x:
            # Fallback: pick best single cutpoint by gain
            best_idx = np.argmax(gains_sorted)
            return [thresh_sorted[best_idx]]

        return [lower, upper]

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.feature_names = feature_names
        self.cutpoints = {}
        cut_id = 0

        for feat_idx in range(X.shape[1]):
            x_feat = X[:, feat_idx].astype(float)
            if np.std(x_feat) == 0:
                continue

            thresholds, gains = self._tree_thresholds(x_feat, y)
            pruned_t, pruned_g = self._prune_thresholds(x_feat, thresholds, gains)
            selected = self._select_cutpoints(pruned_t, pruned_g, x_feat)

            for cp in selected:
                self.cutpoints[cut_id] = (feat_idx, cp)
                cut_id += 1

        return self

    def transform(self, X):
        X = np.asarray(X)
        if len(self.cutpoints) == 0:
            return np.zeros((X.shape[0], 0), dtype=int)

        Xbin = []
        for feat_idx, thresh in self.cutpoints.values():
            col = (X[:, feat_idx] <= thresh).astype(int)
            Xbin.append(col.reshape(-1, 1))

        return np.hstack(Xbin)

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names=feature_names).transform(X)