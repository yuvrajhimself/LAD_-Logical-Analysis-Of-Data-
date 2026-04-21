import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

class DecisionTreeCutpointBinarizerV2:
    """
    Ultra-permissive version of DecisionTreeCutpointBinarizer.
    Designed to generate HUNDREDS of high-quality binary features per numeric column.
    Ideal for: LightGBM/XGBoost with lots of features, RuleFit, or neural nets on binarized inputs.
    """
    def __init__(self,
                 mode="dense",           # "one", "two", "all", "dense", "greedy"
                 max_depth=8,
                 min_samples_leaf=5,
                 min_support=5,          # per side
                 top_k_per_feature=20,      # only used in "greedy" mode
                 edge_fraction=0.0,      # keep edges!
                 min_interval_fraction=0.02,
                 min_impurity_decrease=0.0,
                 random_state=42):

        valid_modes = ("one", "two", "all", "dense", "greedy")
        assert mode in valid_modes, f"mode must be one of {valid_modes}"
        self.mode = mode

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_support = min_support
        self.edge_fraction = edge_fraction
        self.min_interval_fraction = min_interval_fraction
        self.top_k_per_feature = top_k_per_feature
        self.random_state = random_state

        self.cutpoints = {}  # cut_id → (feat_idx, threshold)
        self.feature_names = None

    def print_cutpoints_readable(self):
        print("\n=== AGGRESSIVE BINARIZER – DISCOVERED CUTPOINTS ===")
        if not self.cutpoints:
            print("No cutpoints found!")
            return
        for cut_id, (feat_idx, thresh) in self.cutpoints.items():
            name = self.feature_names[feat_idx] if self.feature_names else f"f{feat_idx}"
            print(f"Cut {cut_id:4d}:  {name} <= {thresh:.6f}")
        print(f"Total binary features: {len(self.cutpoints)}\n")

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
        thresholds = np.unique(np.round(thresholds, 10))  # higher precision

        if len(thresholds) == 0:
            return np.array([]), np.array([])

        # Compute actual information gain for each threshold
        gains = []
        for t in thresholds:
            left = x.flatten() <= t
            n_l, n_r = left.sum(), (~left).sum()
            if n_l < 2 or n_r < 2:
                gains.append(0.0)
                continue
            p_l = y[left].mean()
            p_r = y[~left].mean()
            p = y.mean()
            gain = (n_l + n_r) * (
                p*(1-p) - (n_l/(n_l+n_r))*p_l*(1-p_l) - (n_r/(n_l+n_r))*p_r*(1-p_r)
            )
            gains.append(gain)

        order = np.argsort(thresholds)
        return thresholds[order], np.array(gains)[order]

    def _light_prune(self, x, thresholds, gains):
        """Only removes cuts with almost no support on either side"""
        if len(thresholds) == 0:
            return [], []

        kept_t, kept_g = [], []
        for t, g in zip(thresholds, gains):
            left_count = np.sum(x <= t)
            right_count = len(x) - left_count
            if left_count >= self.min_support and right_count >= self.min_support:
                kept_t.append(t)
                kept_g.append(g)
        return kept_t, kept_g

    def _select_cutpoints(self, pruned_thresh, pruned_gains, x):
        if len(pruned_thresh) == 0:
            return []

        arr_t = np.array(pruned_thresh)
        arr_g = np.array(pruned_gains)
        sorted_idx = np.argsort(arr_t)
        t_sorted = arr_t[sorted_idx]
        g_sorted = arr_g[sorted_idx]

        if self.mode == "one":
            return [t_sorted[np.argmax(g_sorted)]]

        elif self.mode == "two":
            if len(t_sorted) == 1:
                return [t_sorted[0]]
            lower, upper = t_sorted[0], t_sorted[-1]
            if (upper - lower) < self.min_interval_fraction * np.ptp(x):
                return [t_sorted[np.argmax(g_sorted)]]
            return [lower, upper]

        elif self.mode in ("all", "dense"):
            return t_sorted.tolist()

        elif self.mode == "greedy":
            if len(g_sorted) <= self.top_k_per_feature:
                return t_sorted.tolist()
            top_idx = np.argsort(g_sorted)[-self.top_k_per_feature:][::-1]
            return t_sorted[top_idx].tolist()

        return []

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.feature_names = feature_names
        self.cutpoints = {}
        cut_id = 0

        for feat_idx in range(X.shape[1]):
            x = X[:, feat_idx].astype(float)
            if np.std(x) == 0:
                continue

            raw_thresh, raw_gains = self._tree_thresholds(x, y)
            pruned_thresh, pruned_gains = self._light_prune(x, raw_thresh, raw_gains)

            selected = self._select_cutpoints(pruned_thresh, pruned_gains, x)

            for thresh in selected:
                self.cutpoints[cut_id] = (feat_idx, thresh)
                cut_id += 1

        return self

    def transform(self, X):
        X = np.asarray(X)
        if not self.cutpoints:
            return np.zeros((X.shape[0], 0), dtype=np.int8)

        bins = []
        for feat_idx, thresh in self.cutpoints.values():
            bins.append((X[:, feat_idx] <= thresh).astype(np.int8))
        return np.column_stack(bins)

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names).transform(X)