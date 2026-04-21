import numpy as np
import itertools


class Eager:
    """
    Sequential Covering Rule Learner for LAD.

    Replaces the exhaustive enumerate-then-filter approach with a
    covering algorithm: mine one rule at a time, remove the rows it
    covers, repeat on the remainder. This guarantees:

      - Each training row contributes to exactly one rule.
      - Rule count is bounded by n_rows / min_rule_support.
      - No two rules overlap, so there is no rule explosion.

    Rule search at each step:
      For each subset size 1..n_features (smallest first), evaluate
      every feature combination and every unique binary pattern over
      those features on the currently active rows. Keep the candidate
      with the highest purity (ties broken by coverage count). Once a
      candidate meeting min_purity is found at the current size, stop
      expanding to larger sizes — this gives the shortest sufficient rule.

    Stopping conditions:
      - Active rows exhausted.
      - No pattern of any size meets min_purity with min_rule_support rows.
      - n_rules_cap reached (safety valve, default 200).
    """

    def __init__(self, binarizer=None, selector=None, purity=0.75,
                 verbose=True, threshold=0, min_rule_support=3,
                 n_rules_cap=200):
        self.min_purity       = purity
        self.verbose          = verbose
        self.binarizer        = binarizer
        self.selector         = selector
        self.rules            = []
        self.threshold        = threshold
        self.min_rule_support = min_rule_support
        self.n_rules_cap      = n_rules_cap

    def _build_bin_names(self, original_feature_names):
        names = []
        for cut_id in self.selector.best_subset:
            if hasattr(self.binarizer, "cutpoint_readable"):
                names.append(self.binarizer.cutpoint_readable(cut_id))
            else:
                feat_idx, thresh = self.binarizer.cutpoints[cut_id]
                names.append(
                    f"{original_feature_names[feat_idx]} <= {thresh:.4f}"
                )
        return names

    def _best_rule_on(self, Xn, y, active_idx, bin_names):
        """
        Search for the best rule over active_idx rows.
        Tries subsets of increasing size; returns the first rule that
        meets min_purity at the smallest size found (shortest-first).
        Within a size, picks highest purity then highest coverage.
        Returns None if no valid rule exists.
        """
        n_features = Xn.shape[1]
        Xa = Xn[active_idx]
        ya = y[active_idx]

        for size in range(1, n_features + 1):
            best = None  # (purity, repet, attrs, values, label)

            for attrs in itertools.combinations(range(n_features), size):
                unique_patterns = np.unique(Xa[:, attrs], axis=0)

                for pattern in unique_patterns:
                    mask        = np.all(Xa[:, attrs] == pattern, axis=1)
                    covered_idx = np.where(mask)[0]
                    repet       = len(covered_idx)

                    if repet < self.min_rule_support:
                        continue
                    if repet <= self.threshold:
                        continue

                    labels, counts = np.unique(ya[covered_idx],
                                               return_counts=True)
                    label  = labels[np.argmax(counts)]
                    purity = counts.max() / repet

                    if purity < self.min_purity:
                        continue

                    # Prefer higher purity, break ties by higher coverage
                    if (best is None
                            or purity > best[0]
                            or (purity == best[0] and repet > best[1])):
                        best = (purity, repet,
                                list(attrs), list(pattern), int(label))

            if best is not None:
                purity, repet, attrs, values, label = best
                readable = [
                    bin_names[a] if v == 1 else f"NOT ({bin_names[a]})"
                    for a, v in zip(attrs, values)
                ]
                return {
                    "label":    label,
                    "attrs":    attrs,
                    "values":   values,
                    "purity":   float(purity),
                    "repet":    int(repet),
                    "readable": readable,
                }
            # No valid rule at this size — try larger subsets

        return None  # No rule found at any size

    def fit(self, Xsel, y, original_feature_names):
        self.rules.clear()
        self.original_feature_names = original_feature_names

        Xn         = Xsel.astype(int)
        n_features = Xn.shape[1]
        bin_names  = self._build_bin_names(original_feature_names)

        if self.verbose:
            print("\n=== Eager Sequential Covering Started ===")
            print(f"Selected binary matrix shape: {Xsel.shape}")
            if n_features > 12:
                print(f"[Warning] {n_features} features — inner search is "
                      f"O(2^n) per rule. Consider maxpatterns for large n.")

        # True class counts for weight denominator
        labels_u, counts_u = np.unique(y, return_counts=True)
        label_counts = dict(zip(labels_u.tolist(), counts_u.tolist()))

        active_idx = np.arange(len(y))  # indices of uncovered rows

        while len(active_idx) >= self.min_rule_support:
            if len(self.rules) >= self.n_rules_cap:
                if self.verbose:
                    print(f"  Rule cap ({self.n_rules_cap}) reached. Stopping.")
                break

            rule = self._best_rule_on(Xn, y, active_idx, bin_names)

            if rule is None:
                # No pure rule exists on remaining rows
                if self.verbose:
                    print(f"  No valid rule found on {len(active_idx)} "
                          f"remaining rows. Stopping.")
                break

            # Assign weight using true class count as denominator
            denom          = max(1, label_counts.get(rule["label"], 1))
            rule["weight"] = min(1.0, rule["repet"] / denom)
            self.rules.append(rule)

            # Remove covered rows from active set
            mask_covered = np.all(
                Xn[active_idx][:, rule["attrs"]] == rule["values"],
                axis=1
            )
            n_covered  = mask_covered.sum()
            active_idx = active_idx[~mask_covered]

            if self.verbose:
                print(f"  Rule {len(self.rules):3d}: "
                      f"{' AND '.join(rule['readable'])[:60]} "
                      f"-> {rule['label']} "
                      f"| purity={rule['purity']:.2f} "
                      f"| covered={n_covered} "
                      f"| remaining={len(active_idx)}")

        self.rules.sort(key=lambda x: (-x["weight"], -x["purity"], x["label"]))

        if self.verbose:
            print(f"\nGenerated {len(self.rules)} covering rules "
                  f"({len(active_idx)} rows uncovered -> majority fallback).\n")

    def print_rules(self, top_n=20):
        if not self.rules:
            print("No rules to display.")
            return
        print(f"\n=== TOP {top_n} RULES (sorted by weight) ===")
        for i, r in enumerate(self.rules[:top_n], 1):
            print(f"Rule {i:2d}: IF {' AND '.join(r['readable'])} "
                  f"THEN class = {r['label']} "
                  f"| Purity={r['purity']:.3f} "
                  f"| Support={r['repet']} "
                  f"| Weight={r['weight']:.3f}")