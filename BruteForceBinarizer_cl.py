import numpy as np
import itertools
from sklearn.utils.class_weight import compute_sample_weight


class BruteForceBinarizer:
    """
    Binarizer with separate strategies for numerical and categorical columns.

    Numerical columns
    -----------------
    Brute-force: every midpoint between consecutive distinct values is a
    candidate cutpoint. Each candidate is scored by Gini information gain.
    Candidates that are "too close" are pruned (both distance-based and
    support-based checks). Top-k survivors are kept per column.

    Categorical columns
    -------------------
    Every unique value produces a candidate binary cut  (x == v).
    Every unordered pair of unique values produces a candidate binary cut
    (x == a OR x == b).  All candidates are scored by Gini information gain.
    All are pooled together and top-k are kept per column.

    Output format
    -------------
    Identical to DecisionTreeCutpointBinarizerV2:
        self.cutpoints   : { cut_id -> (feat_idx, threshold) }
                           For numerical: threshold is a float.
                           For categorical singles: threshold stored as a
                           frozenset({value}).
                           For categorical pairs: threshold stored as a
                           frozenset({a, b}).
        self.transform() : returns int8 matrix, 1 where condition is met.

    The readable-string helpers used by selectors and miners are patched to
    handle both numeric and categorical cut representations.

    Parameters
    ----------
    top_k_per_feature     : int   — max cutpoints kept per column (default 20)
    min_support           : int   — min samples required on each side of a cut
    min_interval_fraction : float — min gap between two numeric cuts as a
                                    fraction of the feature's range; cuts
                                    closer than this are merged (best kept)
    categorical_cols      : list  — column indices (0-based) to treat as
                                    categorical; all others treated as numerical
    mode                  : str   — "greedy" uses top_k_per_feature;
                                    "all"/"dense" keeps everything that passes
                                    pruning; "one" keeps single best;
                                    "two" keeps lowest + highest
    """

    def __init__(
        self,
        top_k_per_feature=20,
        min_support=5,
        min_interval_fraction=0.02,
        categorical_cols=None,
        max_group_size=3,
        mode="greedy",
        random_state=42,
    ):
        # max_group_size: maximum subset size for categorical groupings.
        # Size 1 = singles only, 2 = singles + pairs, 3 = + triples, etc.
        # For a column with k unique values, at most min(max_group_size, k-1)
        # is used — the full set is always excluded as it is uninformative.
        # Cleveland has at most 4 unique values per categorical column,
        # so max_group_size=3 exhausts all meaningful subsets.
        valid_modes = ("one", "two", "all", "dense", "greedy")
        assert mode in valid_modes, f"mode must be one of {valid_modes}"

        self.top_k_per_feature     = top_k_per_feature
        self.min_support           = min_support
        self.min_interval_fraction = min_interval_fraction
        self.categorical_cols      = set(categorical_cols) if categorical_cols else set()
        self.max_group_size        = max_group_size
        self.mode                  = mode
        self.random_state          = random_state

        # cut_id -> (feat_idx, threshold)
        # threshold is float for numerical, frozenset for categorical
        self.cutpoints    = {}
        self.feature_names = None

    # ─────────────────────────────────────────────────────────────────────
    # Shared: information gain scorer
    # ─────────────────────────────────────────────────────────────────────

    def _gini_gain(self, mask, y):
        """
        Gini information gain of splitting y by boolean mask.
        Returns 0.0 if either side has fewer than min_support samples.
        """
        n = len(y)
        n_l = mask.sum()
        n_r = n - n_l

        if n_l < self.min_support or n_r < self.min_support:
            return 0.0

        def gini(labels):
            if len(labels) == 0:
                return 0.0
            p = labels.mean()          # works for binary 0/1 labels
            return p * (1.0 - p)

        p_l = n_l / n
        p_r = n_r / n
        gain = gini(y) - p_l * gini(y[mask]) - p_r * gini(y[~mask])
        return float(gain)

    # ─────────────────────────────────────────────────────────────────────
    # Numerical: brute-force midpoints
    # ─────────────────────────────────────────────────────────────────────

    def _numerical_candidates(self, x, y):
        """
        Returns (thresholds, gains) arrays — one entry per surviving candidate.
        Candidates: midpoints between every pair of consecutive distinct values.
        Pruning: drop if support too small on either side OR if gap to the
                 previous kept threshold is less than min_interval_fraction
                 of the feature range.
        """
        unique_vals = np.unique(x)
        if len(unique_vals) < 2:
            return np.array([]), np.array([])

        # All midpoints
        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        # Score every midpoint
        raw = []
        for t in midpoints:
            mask = x <= t
            g = self._gini_gain(mask, y)
            raw.append((t, g))

        if not raw:
            return np.array([]), np.array([])

        raw.sort(key=lambda tg: tg[0])          # sort by threshold value
        thresholds = np.array([tg[0] for tg in raw])
        gains      = np.array([tg[1] for tg in raw])

        # Support check already done inside _gini_gain (gain==0 if violated)
        # Distance-based closeness pruning: walk left to right, drop a cut if
        # it is within min_interval_fraction * range of the previously kept cut
        feat_range = float(x.max() - x.min())
        min_gap    = self.min_interval_fraction * feat_range if feat_range > 0 else 0.0

        kept_t, kept_g = [], []
        last_kept = -np.inf
        for t, g in zip(thresholds, gains):
            if g == 0.0:                          # failed support check
                continue
            if (t - last_kept) < min_gap:         # too close to previous kept
                # Replace with the better one
                if kept_g and g > kept_g[-1]:
                    kept_t[-1] = t
                    kept_g[-1] = g
                    last_kept  = t
                continue
            kept_t.append(t)
            kept_g.append(g)
            last_kept = t

        return np.array(kept_t), np.array(kept_g)

    # ─────────────────────────────────────────────────────────────────────
    # Categorical: all non-empty proper subsets up to max_group_size
    # ─────────────────────────────────────────────────────────────────────

    def _categorical_candidates(self, x, y):
        """
        Returns list of (threshold, gain) where threshold is a frozenset.

        For a column with k unique values, generates every non-empty proper
        subset of size 1 up to min(max_group_size, k-1).
        The full set is excluded — it covers every row so its mask is
        all-ones and carries zero discriminative information.

        Examples for cp with values {1,2,3,4} and max_group_size=3:
          Size 1 (singles) : {1}, {2}, {3}, {4}
          Size 2 (pairs)   : {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4}
          Size 3 (triples) : {1,2,3}, {1,2,4}, {1,3,4}, {2,3,4}
          Size 4 excluded  : {1,2,3,4} -- trivial, always 1

        max_group_size caps the subset size to prevent combinatorial
        explosion on high-cardinality columns. Default 3 safely covers
        all Cleveland categorical columns.
        """
        unique_vals = np.unique(x)
        n_unique    = len(unique_vals)
        candidates  = []

        # Proper subsets only: size 1 up to min(max_group_size, n_unique - 1)
        max_size = min(self.max_group_size, n_unique - 1)

        for size in range(1, max_size + 1):
            for combo in itertools.combinations(unique_vals, size):
                mask = np.isin(x, combo)
                g    = self._gini_gain(mask, y)
                if g > 0.0:
                    candidates.append((frozenset(combo), g))

        # Dedup by actual binary mask: two different frozensets that produce
        # the same 0/1 pattern on the training data are identical cuts.
        # Keep the one with higher Gini gain.
        seen_masks = {}
        for fs, g in candidates:
            mask_key = tuple(np.isin(x, list(fs)).astype(np.int8))
            if mask_key not in seen_masks or g > seen_masks[mask_key][1]:
                seen_masks[mask_key] = (fs, g)

        return list(seen_masks.values())  # list of (frozenset, float)

    # ─────────────────────────────────────────────────────────────────────
    # Mode-based selection (numerical)
    # ─────────────────────────────────────────────────────────────────────

    def _select_numerical(self, thresholds, gains, x):
        """Apply mode-based selection to pruned numerical candidates."""
        if len(thresholds) == 0:
            return []

        order   = np.argsort(thresholds)
        t_s     = thresholds[order]
        g_s     = gains[order]

        if self.mode == "one":
            return [t_s[np.argmax(g_s)]]

        elif self.mode == "two":
            if len(t_s) == 1:
                return [t_s[0]]
            lower, upper = t_s[0], t_s[-1]
            feat_range = float(x.max() - x.min())
            if feat_range > 0 and (upper - lower) < self.min_interval_fraction * feat_range:
                return [t_s[np.argmax(g_s)]]
            return [lower, upper]

        elif self.mode in ("all", "dense"):
            return t_s.tolist()

        elif self.mode == "greedy":
            if len(g_s) <= self.top_k_per_feature:
                return t_s.tolist()
            # Take top-k by gain, then return sorted by threshold value
            top_idx = np.argsort(g_s)[-self.top_k_per_feature:]
            return t_s[np.sort(top_idx)].tolist()

        return []

    # ─────────────────────────────────────────────────────────────────────
    # Mode-based selection (categorical)
    # ─────────────────────────────────────────────────────────────────────

    def _select_categorical(self, candidates):
        """
        candidates: list of (frozenset, gain)
        Apply mode/top-k selection across the combined singles+pairs pool.
        """
        if not candidates:
            return []

        gains = np.array([g for _, g in candidates])

        if self.mode == "one":
            return [candidates[int(np.argmax(gains))][0]]

        elif self.mode == "two":
            if len(candidates) == 1:
                return [candidates[0][0]]
            idx_sorted = np.argsort(gains)
            return [candidates[idx_sorted[0]][0], candidates[idx_sorted[-1]][0]]

        elif self.mode in ("all", "dense"):
            return [fs for fs, _ in candidates]

        elif self.mode == "greedy":
            if len(candidates) <= self.top_k_per_feature:
                return [fs for fs, _ in candidates]
            top_idx = np.argsort(gains)[-self.top_k_per_feature:]
            return [candidates[i][0] for i in top_idx]

        return []

    # ─────────────────────────────────────────────────────────────────────
    # fit / transform / fit_transform
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.feature_names = feature_names
        self.cutpoints = {}
        cut_id = 0

        for feat_idx in range(X.shape[1]):
            x = X[:, feat_idx]

            if feat_idx in self.categorical_cols:
                # ── Categorical path ─────────────────────────────────
                candidates = self._categorical_candidates(x, y)
                selected   = self._select_categorical(candidates)
                for fs in selected:
                    self.cutpoints[cut_id] = (feat_idx, fs)
                    cut_id += 1

            else:
                # ── Numerical path ───────────────────────────────────
                if np.std(x) == 0:
                    continue
                thresholds, gains = self._numerical_candidates(x, y)
                selected = self._select_numerical(thresholds, gains, x)
                for thresh in selected:
                    self.cutpoints[cut_id] = (feat_idx, thresh)
                    cut_id += 1

        # Final global dedup: remove any cutpoints whose binary column on
        # the training data is identical to an already-kept cutpoint.
        # Keeps the first occurrence (lowest cut_id) for each unique column.
        seen_cols = {}
        to_remove = []
        X_arr = np.asarray(X, dtype=float)
        for cid, (fi, thresh) in self.cutpoints.items():
            xc = X_arr[:, fi]
            if isinstance(thresh, frozenset):
                col = np.zeros(len(xc), dtype=np.int8)
                for v in thresh:
                    col |= (xc == v).astype(np.int8)
            else:
                col = (xc <= thresh).astype(np.int8)
            key = col.tobytes()
            if key in seen_cols:
                to_remove.append(cid)
            else:
                seen_cols[key] = cid
        for cid in to_remove:
            del self.cutpoints[cid]
        # Re-index cut_ids to be contiguous starting from 0
        self.cutpoints = {
            new_id: val
            for new_id, val in enumerate(self.cutpoints.values())
        }

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.cutpoints:
            return np.zeros((X.shape[0], 0), dtype=np.int8)

        cols = []
        for feat_idx, threshold in self.cutpoints.values():
            x = X[:, feat_idx]
            if isinstance(threshold, frozenset):
                # Categorical: 1 if value is in the set
                col = np.zeros(len(x), dtype=np.int8)
                for v in threshold:
                    col |= (x == v).astype(np.int8)
                cols.append(col)
            else:
                # Numerical: 1 if x <= threshold
                cols.append((x <= threshold).astype(np.int8))

        return np.column_stack(cols)

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names).transform(X)

    # ─────────────────────────────────────────────────────────────────────
    # Readable helpers  (used by selectors and miners)
    # ─────────────────────────────────────────────────────────────────────

    def cutpoint_readable(self, cut_id):
        """Return a human-readable string for a given cut_id."""
        feat_idx, threshold = self.cutpoints[cut_id]
        name = (self.feature_names[feat_idx]
                if self.feature_names is not None
                else f"f{feat_idx}")
        if isinstance(threshold, frozenset):
            vals = sorted(threshold)
            if len(vals) == 1:
                return f"{name} == {vals[0]}"
            else:
                vals_str = ", ".join(str(v) for v in vals)
                return f"{name} IN {{{vals_str}}}"
        else:
            return f"{name} <= {threshold:.4f}"

    def print_cutpoints_readable(self):
        print("\n=== BruteForceBinarizer — DISCOVERED CUTPOINTS ===")
        if not self.cutpoints:
            print("No cutpoints found.")
            return
        for cut_id, (feat_idx, threshold) in self.cutpoints.items():
            print(f"  Cut {cut_id:4d}:  {self.cutpoint_readable(cut_id)}")
        print(f"Total binary features: {len(self.cutpoints)}\n")