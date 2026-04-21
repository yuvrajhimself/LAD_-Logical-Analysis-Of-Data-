import numpy as np


class LazyPatterns:
    """
    Lazy Rule Learner — truly lazy, no rules mined upfront.

    fit():
        Stores training data and computes a per-feature purity weight.
        Purity weight for feature f = max purity achieved by f alone
        across both its values (0 and 1). Higher weight = more discriminating
        = removed last during pruning.

    predict_single():
        Starts with all selected features active for the test instance.
        Checks if training rows matching on all active features are pure
        enough and numerous enough — if yes, majority vote among matches.
        If not, removes the lowest-weight feature and tries again.
        Repeats until a match is found or all features are exhausted.
        Falls back to majority class only as last resort.

    The "rule" for each instance is the subset of its own features
    that survived pruning before a match was found.
    """

    def __init__(self, binarizer, selector, purity=0.75, min_support=3, verbose=False):
        self.binarizer   = binarizer
        self.selector    = selector
        self.purity      = purity
        self.min_support = min_support
        self.verbose     = verbose

    def fit(self, X_train_sel, y_train, original_feature_names):
        self.X_train_sel = X_train_sel.astype(np.int8)
        self.y_train     = y_train.astype(np.int8)
        n_features       = X_train_sel.shape[1]

        # Build readable names for selected features.
        # Uses cutpoint_readable() when available (BruteForceBinarizer — handles
        # both numeric and categorical cuts). Falls back to legacy numeric format
        # for DecisionTreeCutpointBinarizerV2.
        selected_cut_ids = self.selector.best_subset
        self.bin_names   = []
        for cut_id in selected_cut_ids:
            if hasattr(self.binarizer, "cutpoint_readable"):
                self.bin_names.append(self.binarizer.cutpoint_readable(cut_id))
            else:
                feat_idx, thresh = self.binarizer.cutpoints[cut_id]
                self.bin_names.append(
                    f"{original_feature_names[feat_idx]} <= {thresh:.4f}"
                )

        # ── Per-feature purity weight ────────────────────────────────────────
        # For each feature, check both values (0 and 1).
        # Purity for a value v = fraction of rows where feature=v that share
        # the majority label among those rows.
        # Feature weight = max purity across both values.
        # A weight of 1.0 means one value of this feature perfectly separates classes.
        self.feature_weights = np.zeros(n_features, dtype=float)

        for f in range(n_features):
            best_purity = 0.0
            for val in (0, 1):
                mask    = self.X_train_sel[:, f] == val
                covered = self.y_train[mask]
                if len(covered) < self.min_support:
                    continue
                counts  = np.bincount(covered.astype(int),
                                      minlength=int(self.y_train.max()) + 1)
                pur     = counts.max() / len(covered)
                if pur > best_purity:
                    best_purity = pur
            self.feature_weights[f] = best_purity

        # Pruning order: ascending weight = least discriminating removed first
        # Pre-compute once so every predict_single uses the same order
        self.prune_order = np.argsort(self.feature_weights).tolist()  # low → high

        if self.verbose:
            print(f"[LazyPatterns] fit() — {self.X_train_sel.shape[0]} training rows, "
                  f"{n_features} selected features.")
            print(f"[LazyPatterns] purity={self.purity}, min_support={self.min_support}")
            print(f"[LazyPatterns] Feature weights (purity-based):")
            for f in np.argsort(self.feature_weights)[::-1]:
                print(f"  [{f:3d}] {self.bin_names[f]:50s}  weight={self.feature_weights[f]:.4f}")

        return self

    def _make_readable(self, row_bin, active_cols):
        conds = []
        for i in active_cols:
            if row_bin[i] == 1:
                conds.append(self.bin_names[i])
            else:
                conds.append(f"NOT {self.bin_names[i]}")
        return conds

    def _check_match(self, test_row_sel, active_cols):
        """
        Among training rows that match test_row_sel on all active_cols,
        check if there are enough (>= min_support) and they are pure enough
        (>= purity). Returns (matched, label, purity, support) if found,
        else (False, None, 0, 0).
        """
        if not active_cols:
            return False, None, 0.0, 0

        vals    = [int(test_row_sel[c]) for c in active_cols]
        mask    = np.all(self.X_train_sel[:, active_cols] == vals, axis=1)
        covered = np.where(mask)[0]

        if len(covered) < self.min_support:
            return False, None, 0.0, len(covered)

        y_cov   = self.y_train[covered]
        counts  = np.bincount(y_cov.astype(int),
                               minlength=int(self.y_train.max()) + 1)
        pur     = counts.max() / len(covered)

        if pur < self.purity:
            return False, None, pur, len(covered)

        label = int(np.argmax(counts))
        return True, label, float(pur), len(covered)

    def predict_single(self, test_row_sel):
        """
        Prune features from lowest to highest weight until a pure match is found.
        Returns a dict with: label, purity, support, exact_match, stage, rule.
        """
        # Start with all features active
        active_set = list(range(len(test_row_sel)))

        # Pruning queue: order in which features will be removed if needed
        # (lowest weight first — least discriminating dropped first)
        prune_queue = [f for f in self.prune_order if f in active_set]

        # ── Try full feature set first (exact match equivalent) ──────────────
        matched, label, pur, sup = self._check_match(test_row_sel, active_set)
        if matched:
            return {
                "label":       label,
                "purity":      pur,
                "support":     sup,
                "exact_match": True,
                "stage":       "exact",
                "rule":        self._make_readable(test_row_sel, active_set)
            }

        # ── Prune one feature at a time ──────────────────────────────────────
        for feat_to_remove in prune_queue:
            active_set.remove(feat_to_remove)

            if not active_set:
                break

            matched, label, pur, sup = self._check_match(test_row_sel, active_set)
            if matched:
                return {
                    "label":       label,
                    "purity":      pur,
                    "support":     sup,
                    "exact_match": False,
                    "stage":       "pruned",
                    "rule":        self._make_readable(test_row_sel, active_set)
                }

        # ── Majority class fallback ──────────────────────────────────────────
        maj = int(np.argmax(np.bincount(self.y_train.astype(int))))
        return {
            "label":       maj,
            "purity":      0.5,
            "support":     len(self.y_train),
            "exact_match": False,
            "stage":       "fallback",
            "rule":        ["<majority class fallback>"]
        }

    def predict(self, X_test_sel):
        return [self.predict_single(row) for row in X_test_sel]