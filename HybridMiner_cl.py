"""
HybridMiner
===========
Runs a primary miner (MaxPatterns or Eager) then identifies training rows
that are not covered by any rule it produced. Runs the secondary miner
on those uncovered rows to generate additional rules.

The combined rule set covers more training rows than either miner alone.

Primary/secondary choice:
  - If primary is Eager (sequential covering), uncovered rows are exactly
    what Eager left behind when no pure rule could be found.
  - If primary is MaxPatterns, uncovered rows are those matched by no rule
    at all (not even a low-purity one — MaxPatterns may leave some rows
    uncovered when every pattern containing them is impure).

Usage in main.py: set PATTERN_MINER = "hybrid_mp_eager" or "hybrid_eager_mp".
"""

import numpy as np


class HybridMiner:
    """
    Wraps a primary + secondary miner.
    Exposes .rules (combined), .binarizer, .selector for compatibility
    with all classifiers and evaluate functions.
    """

    def __init__(self, primary, secondary, verbose=True):
        """
        primary   : already-constructed but not yet fitted miner (MaxPatterns or Eager)
        secondary : already-constructed but not yet fitted miner (the other one)
        verbose   : print progress
        """
        self.primary    = primary
        self.secondary  = secondary
        self.verbose    = verbose
        self.rules      = []

        # Expose binarizer/selector from primary for compatibility
        self.binarizer  = primary.binarizer
        self.selector   = primary.selector
        self.min_purity = primary.min_purity

    def fit(self, Xsel, y, original_feature_names):
        Xn = Xsel.astype(int)

        # ── Step 1: fit primary ───────────────────────────────────────────────
        if self.verbose:
            print("\n=== HybridMiner: Primary miner ===")
        self.primary.fit(Xsel, y, original_feature_names)
        primary_rules = self.primary.rules

        if self.verbose:
            print(f"  Primary produced {len(primary_rules)} rules.")

        # ── Step 2: find uncovered training rows ──────────────────────────────
        covered = np.zeros(len(y), dtype=bool)
        for rule in primary_rules:
            mask     = np.all(Xn[:, rule["attrs"]] == rule["values"], axis=1)
            covered |= mask

        uncovered_idx = np.where(~covered)[0]

        if self.verbose:
            print(f"  Uncovered rows after primary: "
                  f"{len(uncovered_idx)}/{len(y)}")

        # ── Step 3: fit secondary on uncovered rows ───────────────────────────
        secondary_rules = []
        if len(uncovered_idx) >= getattr(self.secondary, "min_rule_support", 3):
            Xsel_residual = Xsel[uncovered_idx]
            y_residual    = y[uncovered_idx]

            if self.verbose:
                print(f"  Running secondary miner on {len(uncovered_idx)} "
                      f"uncovered rows...")

            self.secondary.fit(Xsel_residual, y_residual,
                               original_feature_names)
            secondary_rules = self.secondary.rules

            if self.verbose:
                print(f"  Secondary produced {len(secondary_rules)} "
                      f"additional rules.")
        else:
            if self.verbose:
                print(f"  Too few uncovered rows "
                      f"({len(uncovered_idx)}) for secondary miner. Skipping.")

        # ── Step 4: combine and sort ──────────────────────────────────────────
        # Mark each rule's origin for transparency
        for r in primary_rules:
            r.setdefault("source", "primary")
        for r in secondary_rules:
            r.setdefault("source", "secondary")

        self.rules = sorted(
            primary_rules + secondary_rules,
            key=lambda r: (-r["weight"], -r["purity"], r["label"])
        )

        if self.verbose:
            n_pri = len(primary_rules)
            n_sec = len(secondary_rules)
            print(f"\nHybridMiner complete: "
                  f"{n_pri} primary + {n_sec} secondary = "
                  f"{len(self.rules)} total rules.\n")

    def print_rules(self, top_n=20):
        if not self.rules:
            print("No rules to display.")
            return
        print(f"\n=== TOP {top_n} HYBRID RULES (sorted by weight) ===")
        for i, r in enumerate(self.rules[:top_n], 1):
            src = r.get("source", "?")
            print(f"Rule {i:2d} [{src:9s}]: "
                  f"IF {' AND '.join(r['readable'])} "
                  f"THEN class = {r['label']} "
                  f"| Purity={r['purity']:.3f} "
                  f"| Support={r['repet']} "
                  f"| Weight={r['weight']:.3f}")