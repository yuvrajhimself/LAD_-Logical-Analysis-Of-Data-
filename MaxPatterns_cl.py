import numpy as np
from tomlkit import key


class MaxPatterns:
    """
    MaxPatterns eager rule miner with optional post-hoc redundancy pruning.

    min_unique_coverage: after sorting rules by weight, a rule is only kept
    if it covers at least this many training rows that are NOT already covered
    by a previously accepted higher-weight rule.

    Setting min_unique_coverage=0  (default) keeps all rules — original behaviour.
    Setting min_unique_coverage=3  prunes rules that add fewer than 3 new rows
    of coverage beyond what higher-weight rules already explain.
    Setting min_unique_coverage=-1 is equivalent to 0 (no pruning).

    This controls overlap without turning MaxPatterns into a covering algorithm
    — rules can still overlap, but heavily redundant ones are removed.
    """

    def __init__(self, binarizer=None, selector=None, purity=0.6,
                 verbose=True, threshold=0, min_unique_coverage=0):
        self.min_purity          = purity
        self.verbose             = verbose
        self.binarizer           = binarizer
        self.selector            = selector
        self.rules               = []
        self.threshold           = threshold
        self.min_unique_coverage = min_unique_coverage

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

    def fit(self, Xsel, y, original_feature_names):
        self.rules.clear()
        self.original_feature_names = original_feature_names

        if self.verbose:
            print("\n=== MaxPatterns Rule Mining Started ===")
            print(f"Selected binary matrix shape: {Xsel.shape}")

        Xn        = Xsel.astype(int)
        bin_names = self._build_bin_names(original_feature_names)

        # True class counts as weight denominator — prevents inflation
        # when rules overlap and multiple rules cover the same rows.
        labels_unique, counts_unique = np.unique(y, return_counts=True)
        label_counts = dict(zip(labels_unique.tolist(), counts_unique.tolist()))

        rules_raw = []

        for inst in np.unique(Xn, axis=0):
            attrs = list(range(len(inst)))
            repet, _, purity, label, _ = self._stats(Xn, y, inst, attrs)

            # Greedy attribute deletion: remove conditions one at a time
            # as long as purity stays at or above the threshold
            while len(attrs) > 1 and purity >= self.min_purity:
                best_remove = None
                best_purity = purity

                for a in attrs:
                    trial_attrs = [t for t in attrs if t != a]
                    _, _, p2, _, _ = self._stats(Xn, y, inst, trial_attrs)
                    if p2 >= self.min_purity and p2 >= best_purity:
                        best_purity = p2
                        best_remove = a

                if best_remove is None:
                    break
                attrs.remove(best_remove)
                repet, _, purity, label, _ = self._stats(Xn, y, inst, attrs)

            if purity < self.min_purity or len(attrs) == 0:
                continue

            mask = np.all(Xn[:, attrs] == [int(inst[a]) for a in attrs], axis=1)
            rows = set(np.where(mask)[0])

            rule = {
                "label":  int(label),
                "attrs":  attrs.copy(),
                "values": [int(inst[a]) for a in attrs],
                "purity": float(purity),
                "rows": rows,                      # ✅ store rows instead of repet
                "readable": [
                    bin_names[i] if v == 1 else f"NOT ({bin_names[i]})"
                    for i, v in zip(attrs, [int(inst[a]) for a in attrs])
                ]
            }

            if len(rule["rows"]) > self.threshold:
                rules_raw.append(rule)

        # Dedup by (label, attrs, values) — accumulate support for merged rules
        seen = {}
        for r in rules_raw:
            key = (r["label"], tuple(r["attrs"]), tuple(r["values"]))
            if key not in seen:
                seen[key] = r
            else:
                seen[key]["rows"] |= r["rows"]   # ✅ union

        # Assign weights using true class counts as denominator
        candidates = []
        for r in seen.values():
            r["repet"] = len(r["rows"])   # ✅ correct support

            denom = max(1, label_counts.get(r["label"], 1))
            r["weight"] = min(1.0, r["repet"] / denom)

            candidates.append(r)

        candidates.sort(key=lambda x: (-x["weight"], -x["purity"], x["label"]))

        # ── Redundancy pruning ────────────────────────────────────────────
        # Walk rules in descending weight order. Only keep a rule if it
        # covers at least min_unique_coverage rows not already explained
        # by a previously accepted rule.
        if self.min_unique_coverage > 0:
            already_covered = set()
            pruned = []
            for r in candidates:
                mask = np.all(Xn[:, r["attrs"]] == r["values"], axis=1)
                rows = set(np.where(mask)[0])
                unique_rows = rows - already_covered
                if len(unique_rows) >= self.min_unique_coverage:
                    pruned.append(r)
                    already_covered |= rows
            self.rules = pruned
            if self.verbose:
                print(f"Generated {len(candidates)} rules before pruning, "
                      f"{len(self.rules)} after redundancy pruning "
                      f"(min_unique_coverage={self.min_unique_coverage}).\n")
        else:
            self.rules = candidates
            if self.verbose:
                print(f"Generated {len(self.rules)} high-quality rules.\n")

    def _stats(self, Xn, y, inst, attrs):
        if len(attrs) == 0:
            return 0, 0, 0.0, 0, 1.0
        mask    = np.all(Xn[:, attrs] == [inst[a] for a in attrs], axis=1)
        covered = np.where(mask)[0]
        repet   = len(covered)
        if repet == 0:
            return 0, 0, 0.0, 0, 1.0
        labels, counts = np.unique(y[covered], return_counts=True)
        label  = labels[np.argmax(counts)]
        purity = counts.max() / repet
        return repet, counts.max(), purity, int(label), 0.0

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