"""
LAD Heart Disease Pipeline
==========================
Configure your choices below, then run. All pipeline logic lives in functions.
"""

# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH    = "Heart_disease_cleveland_new.csv"
TEST_SIZE    = 0.25
RANDOM_STATE = 42

# ── Binarizer ────────────────────────────────────────────────────────────────
# "bruteforce" | "decisiontree"
BINARIZER_TYPE = "bruteforce"

# BruteForceBinarizer  (BINARIZER_TYPE = "bruteforce")
BF_MODE              = "greedy"
BF_TOP_K             = 20
BF_MIN_SUPPORT       = 5
BF_MIN_INTERVAL_FRAC = 0.05
# Cleveland categorical column indices (0-based):
#   1=cp, 2=fbs, 5=restecg, 6=exang, 8=slope, 10=ca, 11=thal
BF_CATEGORICAL_COLS  = [1, 2, 5, 6, 8, 10, 11]
# Maximum subset size for categorical groupings.
# 1=singles only | 2=+pairs | 3=+triples
# Cleveland has at most 4 unique values per categorical column;
# max_group_size=3 exhausts all meaningful subsets.
BF_MAX_GROUP_SIZE    = 6

# DecisionTreeCutpointBinarizerV2  (BINARIZER_TYPE = "decisiontree")
DT_MODE        = "greedy"
DT_MAX_DEPTH   = 4
DT_MIN_SAMPLES = 10

# ── Feature Selector ─────────────────────────────────────────────────────────
# "greedy" | "astar" | "mutualinfo" | "mutualinfo_astar"
SELECTOR = "mutualinfo_astar"

# A* heuristic weight — applies to both astar and mutualinfo_astar.
# f = g + ASTAR_H_WEIGHT * h  where h = 1 - lad_score
# Higher = prioritises purity over subset size.
# Set TUNE_ASTAR = True to auto-select from ASTAR_H_WEIGHT_CANDIDATES.
ASTAR_H_WEIGHT            = 5
TUNE_ASTAR                = False
ASTAR_H_WEIGHT_CANDIDATES = [1, 2, 5, 10]
ASTAR_TUNE_EXPANSIONS     = 10000   # expansions budget per candidate during tuning

# ── Pattern Miner ────────────────────────────────────────────────────────────
# "maxpatterns" | "eager" | "genetic" | "lazy"
# "hybrid_mp_eager" | "hybrid_eager_mp"
PATTERN_MINER = "maxpatterns"

# Purity for eager miners (maxpatterns / eager / genetic).
# Set TUNE_PURITY = True to auto-select from PURITY_CANDIDATES via 5-fold CV.
PURITY                = 0.95
TUNE_PURITY           = False
PURITY_CANDIDATES     = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

THRESHOLD = 3
# MaxPatterns redundancy pruning: a rule is only kept if it covers
# at least this many rows NOT already covered by higher-weight rules.
# 0 = no pruning (keep all rules, original behaviour).
# 3-5 = recommended to remove heavily overlapping redundant rules.
MAX_PATTERNS_MIN_UNIQUE_COV = 2
# Minimum training rows a rule must cover to be kept (Eager only).
# Prevents single- and double-row overfit rules from inflating rule count.
# Independent from THRESHOLD — raise this if rule count is too high.
EAGER_MIN_RULE_SUPPORT = 3
# Hard cap on number of rules Eager will produce (safety valve).
EAGER_N_RULES_CAP      = 100

# Genetic algorithm settings
GA_GENERATIONS  = 300
GA_POP_SIZE     = 200
GA_SHARING_SIGMA= 0.3   # Jaccard similarity threshold for fitness sharing; 0 = disabled

# Lazy settings  (PATTERN_MINER = "lazy")
# Separate from PURITY — 1.0 is too strict for lazy. Range: 0.65–0.80
LAZY_PURITY      = 0.75
LAZY_MIN_SUPPORT = 3

# ── Prediction ───────────────────────────────────────────────────────────────
# Minimum rule weight to consider in partial-match prediction.
# Rules below this threshold are skipped in the partial-match pass.
MIN_WEIGHT_THRESHOLD = 0.05

# Class weight for disease patterns in LAD scorer.
# 1.0 = symmetric (identical to original scorer).
# 1.5 = disease patterns count 50% more toward the score.
# 2.0 = disease patterns count double.
# Values above 1.0 bias feature selection toward subsets that
# better discriminate sick patients, reducing false negatives.
CLASS_WEIGHT_DISEASE = 1

# Partial match scoring mode for best-match classifier.
# "overlap_only"        : rank candidates by overlap fraction only
#                         (weight is a secondary tiebreaker).
# "overlap_times_weight": rank by overlap * weight, giving stronger
#                         rules more influence in the partial pass.
#                         Better for borderline patients.
PARTIAL_MATCH_MODE = "overlap_only"

# Soft voting disease threshold.
# Predict disease only if disease weight / total weight >= this.
# 0.5 = majority rule (default). Higher = more conservative.
# Lower = more aggressive (more disease predictions).
SV_DISEASE_THRESHOLD = 0.4

# Overlapping rule set for voting classifiers.
# When True, classifiers 2/3/4 (soft vote, assume healthy, assume sick)
# use a separate MaxPatterns rule set mined WITHOUT row removal
# (full overlap allowed) so every pattern gets a voice.
# Classifier 1 (best match) always uses the primary covering rules.
# Only applies when PATTERN_MINER is "maxpatterns", "eager",
# "hybrid_mp_eager", or "hybrid_eager_mp".
USE_OVERLAPPING_RULES_FOR_VOTING = True
# Minimum rows an overlapping rule must cover on the full training
# set to be included in the voting rule set.
OVERLAPPING_MIN_COV = 3

# ── Cross-validation ─────────────────────────────────────────────────────────
# Set True to run 5-fold CV after the main evaluation.
RUN_CV    = False
CV_SPLITS = 5

# ================================================================
# IMPORTS
# ================================================================

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)

from MaxPatterns_cl import MaxPatterns
from Eager_cl import Eager
from GeneticRuleMiner_cl import GeneticRuleMiner
from LazyPatterns_cl import LazyPatterns
from AStarFeatureSelector_cl import AStarFeatureSelector
from GreedyLADSelector_cl import GreedyLADSelector
from MutualInfoGreedySelector_cl import MutualInfoGreedySelector
from MutualInfoAStarSelector_cl import MutualInfoAStarSelector
from HybridMiner_cl import HybridMiner
from BruteForceBinarizer_cl import BruteForceBinarizer
from DecisionTreeCutpointBinarizerV2 import DecisionTreeCutpointBinarizerV2
from LADScorer_cl import lad_score

# ================================================================
# PIPELINE FUNCTIONS
# ================================================================

def load_and_split_data(path, test_size, random_state):
    df = pd.read_csv(path)
    print("Dataset loaded:", df.shape)
    print("Features:", df.columns[:-1].tolist())
    print("Target distribution (full):")
    print(df.iloc[:, -1].value_counts(normalize=True).sort_index())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    feature_names = df.columns[:-1].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n" + "="*60)
    print("STRATIFIED SPLIT COMPLETED")
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")
    print("  Train class distribution:")
    for cls, frac in pd.Series(y_train).value_counts(
            normalize=True).sort_index().items():
        print(f"    {cls}: {frac:.3f}")
    print("  Test class distribution:")
    for cls, frac in pd.Series(y_test).value_counts(
            normalize=True).sort_index().items():
        print(f"    {cls}: {frac:.3f}")
    print("="*60 + "\n")

    return X_train, X_test, y_train, y_test, feature_names


def binarize(X_train, y_train, X_test, feature_names,
             binarizer_type,
             bf_mode, bf_top_k, bf_min_support,
             bf_min_interval_frac, bf_categorical_cols,
             bf_max_group_size,
             dt_mode, dt_max_depth, dt_min_samples):
    if binarizer_type == "bruteforce":
        binarizer = BruteForceBinarizer(
            mode=bf_mode,
            top_k_per_feature=bf_top_k,
            min_support=bf_min_support,
            min_interval_fraction=bf_min_interval_frac,
            categorical_cols=bf_categorical_cols,
            max_group_size=bf_max_group_size,
        )
    elif binarizer_type == "decisiontree":
        binarizer = DecisionTreeCutpointBinarizerV2(
            mode=dt_mode,
            max_depth=dt_max_depth,
            min_samples_leaf=dt_min_samples,
        )
    else:
        raise ValueError(
            f"Unknown binarizer_type: '{binarizer_type}'. "
            f"Choose from: bruteforce, decisiontree"
        )

    Xbin_train = binarizer.fit_transform(
        X_train, y_train, feature_names=feature_names
    )
    binarizer.print_cutpoints_readable()

    if hasattr(binarizer, "cutpoint_readable"):
        bin_feature_names = [
            binarizer.cutpoint_readable(cut_id)
            for cut_id in binarizer.cutpoints
        ]
    else:
        bin_feature_names = [
            f"{feature_names[f_idx]} <= {thr:.4f}"
            for cut_id, (f_idx, thr) in binarizer.cutpoints.items()
        ]

    print(f"Binarized training matrix : {Xbin_train.shape}")
    print(f"Binary feature names      : {len(bin_feature_names)} columns\n")

    Xbin_test = binarizer.transform(X_test)
    return binarizer, Xbin_train, Xbin_test, bin_feature_names


def tune_astar_weight(Xbin_train, y_train, bin_feature_names,
                      candidates, tune_expansions):
    """
    Run each h_weight candidate with a limited expansion budget.
    Prefers the weight that finds a consistent subset with fewest features.
    Falls back to the one with the highest LAD score if none is consistent.
    """
    print("\n=== Tuning A* h_weight ===")
    best_w       = candidates[0]
    best_n       = float("inf")
    best_score   = -1.0
    best_consist = False

    for w in candidates:
        sel = AStarFeatureSelector(
            max_expansions=tune_expansions, h_weight=w
        )
        sel.fit(Xbin_train, y_train, bin_feature_names)
        is_consistent = sel.best_subset == getattr(sel, "_best_consistent", None) or \
                        hasattr(sel, "y_clean") and np.array_equal(
                            sel.y_clean, y_train)
        n_feats = len(sel.best_subset)
        score   = lad_score(Xbin_train, y_train, sel.best_subset)
        consist = (sel.X_clean is not None and
                   len(sel.X_clean) == len(y_train))
        print(f"  h_weight={w:3d}  n_features={n_feats:3d}  "
              f"lad_score={score:.4f}  consistent={consist}")

        if consist and (not best_consist or n_feats < best_n):
            best_w       = w
            best_n       = n_feats
            best_consist = True
        elif not best_consist and score > best_score:
            best_w     = w
            best_score = score

    print(f"  Selected h_weight = {best_w}\n")
    return best_w


def select_features(selector_name, Xbin_train, y_train, bin_feature_names,
                    h_weight=5, class_weight_disease=1.0):
    if selector_name == "greedy":
        selector = GreedyLADSelector(
            class_weight_disease=class_weight_disease)
        selector.fit(Xbin_train, y_train, bin_feature_names)
    elif selector_name == "astar":
        selector = AStarFeatureSelector(
            h_weight=h_weight,
            class_weight_disease=class_weight_disease)
        selector.fit(Xbin_train, y_train, bin_feature_names)
    elif selector_name == "mutualinfo":
        selector = MutualInfoGreedySelector(
            class_weight_disease=class_weight_disease)
        selector.fit(Xbin_train, y_train)
    elif selector_name == "mutualinfo_astar":
        selector = MutualInfoAStarSelector(
            h_weight=h_weight,
            class_weight_disease=class_weight_disease)
        selector.fit(Xbin_train, y_train, bin_feature_names)
    else:
        raise ValueError(
            f"Unknown selector: '{selector_name}'. "
            f"Choose from: greedy, astar, mutualinfo, mutualinfo_astar"
        )

    Xsel_train    = selector.X_clean[:, selector.best_subset]
    y_train_clean = selector.y_clean

    print(f"\nSelected feature indices : {selector.best_subset}")
    print(f"Total selected           : {len(selector.best_subset)}\n")
    return selector, Xsel_train, y_train_clean


def _make_miner(miner_name, binarizer, selector, purity, threshold,
                ga_generations, ga_pop_size, ga_sharing_sigma, verbose=True,
                **kwargs):
    if miner_name == "maxpatterns":
        return MaxPatterns(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=verbose, threshold=threshold,
            min_unique_coverage=kwargs.get("max_patterns_min_unique_cov", 0),
        )
    elif miner_name == "eager":
        return Eager(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=verbose, threshold=threshold,
            min_rule_support=kwargs.get("eager_min_rule_support", 3),
            n_rules_cap=kwargs.get("eager_n_rules_cap", 100),
        )
    elif miner_name == "genetic":
        return GeneticRuleMiner(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=verbose, threshold=threshold,
            n_generations=ga_generations, pop_size=ga_pop_size,
            sharing_sigma=ga_sharing_sigma,
        )
    elif miner_name == "hybrid_mp_eager":
        primary = MaxPatterns(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=False, threshold=threshold,
            min_unique_coverage=kwargs.get("max_patterns_min_unique_cov", 0),
        )
        secondary = Eager(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=False, threshold=threshold,
            min_rule_support=kwargs.get("eager_min_rule_support", 3),
            n_rules_cap=kwargs.get("eager_n_rules_cap", 100),
        )
        return HybridMiner(primary, secondary, verbose=verbose)
    elif miner_name == "hybrid_eager_mp":
        primary = Eager(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=False, threshold=threshold,
            min_rule_support=kwargs.get("eager_min_rule_support", 3),
            n_rules_cap=kwargs.get("eager_n_rules_cap", 100),
        )
        secondary = MaxPatterns(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=False, threshold=threshold,
            min_unique_coverage=kwargs.get("max_patterns_min_unique_cov", 0),
        )
        return HybridMiner(primary, secondary, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown miner: '{miner_name}'. "
            f"Choose from: maxpatterns, eager, genetic, "
            f"hybrid_mp_eager, hybrid_eager_mp"
        )


def tune_purity(Xsel_train, y_train, binarizer, selector,
                feature_names, miner_name, threshold,
                candidates, ga_generations, ga_pop_size, ga_sharing_sigma,
                n_splits=5):
    """
    5-fold stratified CV over purity candidates.

    Always uses MaxPatterns internally for scoring — it is fast and
    its purity sensitivity is representative of all eager miners.
    Running the full Genetic or Eager miner inside a 5-fold x 7-candidate
    grid would take hours; MaxPatterns gives a good proxy in seconds.
    The winning purity is then applied to the actual chosen miner.

    Returns the candidate with the highest mean CV accuracy.
    """
    print("\n=== Tuning purity threshold ===")
    print(f"  (using MaxPatterns as fast proxy scorer — "
          f"{n_splits} folds x {len(candidates)} candidates)")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    best_purity = candidates[0]
    best_score  = -1.0

    for purity in candidates:
        fold_scores = []
        for tr_idx, va_idx in skf.split(Xsel_train, y_train):
            Xtr, ytr = Xsel_train[tr_idx], y_train[tr_idx]
            Xva, yva = Xsel_train[va_idx], y_train[va_idx]

            # Always use MaxPatterns here — fast and purity-sensitive
            miner = MaxPatterns(
                binarizer=binarizer, selector=selector,
                purity=purity, verbose=False, threshold=threshold,
            )
            miner.fit(Xtr, ytr, original_feature_names=feature_names)

            if not miner.rules:
                fold_scores.append(0.0)
                continue
            preds, _, _ = predict_all(miner, Xva)
            fold_scores.append(accuracy_score(yva, preds))

        mean_score = float(np.mean(fold_scores))
        print(f"  purity={purity:.2f}  cv_acc={mean_score:.4f}")
        if mean_score > best_score:
            best_score  = mean_score
            best_purity = purity

    print(f"  Selected purity = {best_purity}  "
          f"(cv_acc={best_score:.4f})\n")
    return best_purity


def mine_patterns(miner_name, binarizer, selector, Xsel_train, y_train,
                  feature_names, purity, threshold,
                  ga_generations, ga_pop_size, ga_sharing_sigma):
    miner = _make_miner(
        miner_name, binarizer, selector, purity, threshold,
        ga_generations, ga_pop_size, ga_sharing_sigma, verbose=True,
        eager_min_rule_support=EAGER_MIN_RULE_SUPPORT,
        eager_n_rules_cap=EAGER_N_RULES_CAP,
        max_patterns_min_unique_cov=MAX_PATTERNS_MIN_UNIQUE_COV,
    )
    miner.fit(Xsel_train, y_train, original_feature_names=feature_names)

    if miner.rules:
        print(f"\nTOP {len(miner.rules)} RULES DISCOVERED ON TRAINING DATA:")
        miner.print_rules(top_n=len(miner.rules))
    else:
        print("\n[Warning] No rules were generated. "
              "Try lowering PURITY or THRESHOLD in the config.")
    return miner


def _make_hybrid_miner(miner_name, binarizer, selector,
                       purity, threshold, feature_names):
    """Construct but do not fit a hybrid miner for run_pipeline use."""
    if miner_name == "hybrid_mp_eager":
        primary = MaxPatterns(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=True, threshold=threshold,
            min_unique_coverage=MAX_PATTERNS_MIN_UNIQUE_COV,
        )
        secondary = Eager(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=True, threshold=threshold,
            min_rule_support=EAGER_MIN_RULE_SUPPORT,
            n_rules_cap=EAGER_N_RULES_CAP,
        )
        return HybridMiner(primary, secondary, verbose=True)
    elif miner_name == "hybrid_eager_mp":
        primary = Eager(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=True, threshold=threshold,
            min_rule_support=EAGER_MIN_RULE_SUPPORT,
            n_rules_cap=EAGER_N_RULES_CAP,
        )
        secondary = MaxPatterns(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=True, threshold=threshold,
            min_unique_coverage=MAX_PATTERNS_MIN_UNIQUE_COV,
        )
        return HybridMiner(primary, secondary, verbose=True)
    else:
        raise ValueError(f"Not a hybrid miner name: {miner_name}")


# ================================================================
# PREDICTION
# ================================================================

def _predict_row(row, sorted_rules, min_weight_threshold=0.05,
                 partial_match_mode="overlap_times_weight"):
    """
    Pass 1 — exact match: first rule that fires fully wins immediately.

    Pass 2 — partial match: among rules above min_weight_threshold,
             score each rule and pick the highest.
             Scoring depends on partial_match_mode:
               "overlap_only"         -- score = overlap fraction
                                         weight breaks ties only
               "overlap_times_weight" -- score = overlap * rule weight
                                         jointly rewards high-match
                                         AND high-support rules

             "overlap_times_weight" is better for borderline patients
             because a rule with 70% overlap and 0.9 weight (scores 0.63)
             beats one with 80% overlap but 0.2 weight (scores 0.16).
             This directly reduces FN by giving strong disease rules
             more influence even when they don't fire completely.

    Pass 3 — fallback: no rule cleared the weight threshold.
             Use the highest-weight rule regardless.

    Returns (label, is_exact, score).
    """
    # Pass 1 — exact match
    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return rule["label"], True, 1.0

    # Pass 2 — partial match
    best_label, best_score, best_raw_overlap = None, -1.0, -1.0
    for rule in sorted_rules:
        if rule["weight"] < min_weight_threshold:
            continue
        attrs, values = rule["attrs"], rule["values"]
        overlap = sum(
            row[attrs[i]] == values[i] for i in range(len(attrs))
        ) / len(attrs)

        if partial_match_mode == "overlap_times_weight":
            score = overlap * rule["weight"]
        else:  # "overlap_only"
            score = overlap

        if score > best_score:
            best_score       = score
            best_raw_overlap = overlap
            best_label       = rule["label"]

    if best_label is not None:
        return best_label, False, best_raw_overlap

    # Pass 3 — fallback
    if sorted_rules:
        return sorted_rules[0]["label"], False, 0.0

    return None, False, 0.0


def predict_all(miner, X_selected, min_weight_threshold=0.05,
                partial_match_mode="overlap_times_weight"):
    if not miner.rules:
        raise RuntimeError("Miner has no rules. Cannot predict.")
    sorted_rules = sorted(miner.rules, key=lambda r: r["weight"], reverse=True)
    labels, exacts, scores = [], [], []
    for row in X_selected:
        label, exact, score = _predict_row(
            row, sorted_rules, min_weight_threshold, partial_match_mode
        )
        labels.append(label)
        exacts.append(exact)
        scores.append(score)
    return np.array(labels), np.array(exacts), np.array(scores)


def print_predictions(miner, X_test_selected, y_test,
                      min_weight_threshold=0.05):
    print("\n" + "="*70)
    print("LAD PREDICTIONS -- STRONGEST MATCH (sorted by weight)")
    print("="*70)

    if not miner.rules:
        print("[Warning] No rules available. Cannot print predictions.")
        return np.array([]), np.array([]), []

    sorted_rules = sorted(miner.rules, key=lambda r: r["weight"], reverse=True)
    predictions, confidences, justifications = [], [], []

    for row in X_test_selected:
        label, exact, score = _predict_row(
            row, sorted_rules, min_weight_threshold, PARTIAL_MATCH_MODE
        )
        best_rule = next(
            (r for r in sorted_rules
             if r["label"] == label and
             sum(row[r["attrs"][i]] == r["values"][i]
                 for i in range(len(r["attrs"]))) / len(r["attrs"]) == score),
            sorted_rules[0],
        )
        overlap = sum(
            row[best_rule["attrs"][i]] == best_rule["values"][i]
            for i in range(len(best_rule["attrs"]))
        ) / len(best_rule["attrs"])

        predictions.append(label)
        confidences.append(score)
        justifications.append(
            f"Rule weight={best_rule['weight']:.3f}, "
            f"overlap={overlap:.2f} ({overlap*100:.0f}%), "
            f"-> {' AND '.join(best_rule['readable'])}"
        )

    acc = np.mean(np.array(predictions) == y_test)
    print(f"Test Accuracy (strongest match) : {acc:.4f} "
          f"({int(acc*len(y_test))}/{len(y_test)} correct)")
    print(f"Average confidence score        : {np.mean(confidences):.4f}")
    print(f"Coverage                        : 100% ({len(y_test)}/{len(y_test)})")

    print(f"\nFirst {min(100, len(y_test))} test predictions:")
    for i in range(min(100, len(y_test))):
        print(f"  True={y_test[i]:1d} | Pred={predictions[i]:1d} | "
              f"Score={confidences[i]:.3f} | {justifications[i][:90]}...")

    return np.array(predictions), np.array(confidences), justifications


def _build_overlapping_rules(Xsel_train, y_clean, binarizer, selector,
                              purity, threshold, min_cov,
                              original_feature_names):
    """
    Mine a MaxPatterns rule set with full overlap (min_unique_coverage=0)
    on the COMPLETE training set (no rows removed).
    Used by soft-vote and assume classifiers to get full pattern coverage.

    Only rules covering >= min_cov training rows are kept to avoid
    single-row noise rules inflating scores.
    """
    mp = MaxPatterns(
        binarizer           = binarizer,
        selector            = selector,
        purity              = purity,
        verbose             = False,
        threshold           = threshold,
        min_unique_coverage = 0,      # always full overlap for voting
    )
    mp.fit(Xsel_train, y_clean, original_feature_names=original_feature_names)

    # Filter by min_cov (on the full training set, not the active subset)
    Xn = Xsel_train.astype(int)
    kept = []
    for r in mp.rules:
        mask  = np.all(Xn[:, r["attrs"]] == r["values"], axis=1)
        repet = int(mask.sum())
        if repet >= min_cov:
            r["repet"] = repet          # update to full-set count
            kept.append(r)
    return kept


# ================================================================
# EVALUATION
# ================================================================

# ================================================================
# CLASSIFIERS  (all four run together for comparison)
# ================================================================

def _clf_best_match(row, sorted_rules, min_weight_threshold=0.05,
                    partial_match_mode="overlap_times_weight"):
    """
    Classifier 1 — Best Match.
    Pass 1: first exact match wins immediately.
    Pass 2: pick the rule with the best partial match score.
            Score depends on partial_match_mode (see _predict_row).
    Pass 3: fallback to highest-weight rule.
    Returns (label, confidence).
    """
    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return rule["label"], 1.0

    best_label, best_score = None, -1.0
    for rule in sorted_rules:
        if rule["weight"] < min_weight_threshold:
            continue
        attrs, values = rule["attrs"], rule["values"]
        overlap = sum(row[attrs[i]] == values[i]
                      for i in range(len(attrs))) / len(attrs)
        score = (overlap * rule["weight"]
                 if partial_match_mode == "overlap_times_weight"
                 else overlap)
        if score > best_score:
            best_score = score
            best_label = rule["label"]

    if best_label is not None:
        return best_label, best_score
    if sorted_rules:
        return sorted_rules[0]["label"], 0.0
    return None, 0.0


def _clf_soft_vote(row, sorted_rules, sv_disease_threshold=0.5,
                   min_weight_threshold=0.05):
    """
    Classifier 2 — Soft Voting with tunable disease threshold.

    Every rule that fires exactly adds its weight to a per-class score.
    Disease is predicted only if:
        disease_weight / total_weight >= sv_disease_threshold

    sv_disease_threshold=0.5  -- majority rule (default, symmetric)
    sv_disease_threshold=0.6  -- conservative: needs 60% of weight to call disease
    sv_disease_threshold=0.4  -- aggressive: 40% weight is enough to call disease

    Falls back to best-match if no rule fires.
    Returns (label, confidence).
    """
    scores = {}
    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            scores[rule["label"]] = scores.get(rule["label"], 0.0) + rule["weight"]

    if not scores:
        return _clf_best_match(row, sorted_rules, min_weight_threshold)

    total        = sum(scores.values())
    disease_frac = scores.get(1, 0.0) / total if total > 0 else 0.0

    if disease_frac >= sv_disease_threshold:
        label      = 1
        confidence = disease_frac
    else:
        label      = 0
        confidence = 1.0 - disease_frac

    return label, confidence


def _clf_assume_healthy(row, sorted_rules):
    """
    Classifier 3 — Assume Healthy (default class = 0).
    Start with prediction = 0 (no disease).
    Scan disease rules (label=1) only.
    If any disease rule fires exactly, switch prediction to 1.
    Conservative: only diagnoses disease when there is explicit evidence.
    Returns (label, confidence).
    """
    for rule in sorted_rules:
        if rule["label"] != 1:
            continue
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return 1, rule["weight"]
    return 0, 1.0   # confident in default: no disease rule fired


def _clf_assume_sick(row, sorted_rules):
    """
    Classifier 4 — Assume Sick (default class = 1).
    Start with prediction = 1 (disease).
    Scan healthy rules (label=0) only.
    If any healthy rule fires exactly, switch prediction to 0.
    Aggressive: assumes disease unless explicitly cleared.
    Returns (label, confidence).
    """
    for rule in sorted_rules:
        if rule["label"] != 0:
            continue
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return 0, rule["weight"]
    return 1, 1.0   # confident in default: no healthy rule fired


def _run_classifier(clf_fn, sorted_rules, X, extra_kwargs=None):
    """Run a classifier function over all rows in X. Returns (labels, confs)."""
    labels, confs = [], []
    kw = extra_kwargs or {}
    for row in X:
        lbl, conf = clf_fn(row, sorted_rules, **kw)
        labels.append(lbl if lbl is not None else 0)
        confs.append(conf)
    return np.array(labels), np.array(confs)


def _print_classifier_metrics(name, y_true, y_pred, y_conf, width=28):
    """Print one row of classifier metrics."""
    acc   = accuracy_score(y_true, y_pred)
    bal   = balanced_accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_conf)
    except Exception:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"  {name:<{width}} Acc={acc:.4f}  Bal={bal:.4f}  "
          f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
          f"AUC={auc:.4f}  TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d}")


def evaluate_all_classifiers(miner, X_train_sel, y_train,
                              X_test_sel, y_test,
                              model_name, selector_name, miner_name,
                              purity, threshold,
                              min_weight_threshold=0.05,
                              sv_disease_threshold=0.5,
                              overlapping_rules=None):
    """
    Run all four classifiers on both train and test sets and print a
    side-by-side comparison table.

    Classifiers:
      1. Best-match   — uses miner.rules (covering/pruned)
      2. Soft voting  — uses overlapping_rules if provided, else miner.rules
      3. Assume healthy — uses overlapping_rules if provided, else miner.rules
      4. Assume sick    — uses overlapping_rules if provided, else miner.rules

    overlapping_rules : list of rule dicts from _build_overlapping_rules(),
                        or None to use miner.rules for all classifiers.
    sv_disease_threshold : min disease weight fraction for soft vote to
                           call disease (default 0.5 = majority).
    """
    print("=" * 90)
    print("                 LAD EVALUATION  --  ALL CLASSIFIERS")
    print("=" * 90)
    print("PIPELINE SETTINGS")
    print(f"  Selector              : {selector_name}")
    print(f"  Pattern Miner         : {miner_name}")
    print(f"  Purity threshold      : {purity}")
    print(f"  Support threshold     : {threshold}")
    print(f"  Min weight threshold  : {min_weight_threshold}")
    print(f"  Partial match mode    : {PARTIAL_MATCH_MODE}")
    if miner_name == "eager":
        print(f"  Min rule support      : {getattr(miner, 'min_rule_support', '-')}")
    if miner_name == "maxpatterns":
        muc = getattr(miner, 'min_unique_coverage', 0)
        if muc > 0:
            print(f"  Min unique coverage   : {muc}")
    print()

    if not miner.rules:
        print("[Error] No rules available. Cannot evaluate.")
        return

    n_rules     = len(miner.rules)
    n_features  = X_train_sel.shape[1]
    n_cutpoints = len(miner.binarizer.cutpoints)
    model_size_kb = len(pickle.dumps({
        "cutpoints": miner.binarizer.cutpoints,
        "selected":  miner.selector.best_subset,
        "rules":     miner.rules,
    })) / 1024

    print(f"  Model: {model_name}")
    print(f"  Rules: {n_rules}  |  "
          f"Selected features: {n_features}  |  "
          f"Cutpoints: {n_cutpoints}  |  "
          f"Size: {model_size_kb:.1f} KB")
    print()

    # Classifier 1 always uses primary (covering/pruned) rules
    sorted_primary = sorted(miner.rules,
                            key=lambda r: r["weight"], reverse=True)
    pmm = PARTIAL_MATCH_MODE
    # Classifiers 2/3/4 use overlapping rules if provided, else primary
    if overlapping_rules is not None and len(overlapping_rules) > 0:
        sorted_voting = sorted(overlapping_rules,
                               key=lambda r: r["weight"], reverse=True)
        print(f"  Voting rule set      : {len(sorted_voting)} overlapping rules "
              f"(separate from best-match set)")
    else:
        sorted_voting = sorted_primary
        print(f"  Voting rule set      : same as best-match "
              f"({len(sorted_voting)} rules)")
    print(f"  SV disease threshold : {sv_disease_threshold}")
    print()

    mwt = min_weight_threshold

    # ── Compute predictions for all four classifiers ──────────────────────
    # Classifier 1: best-match uses primary rules
    bm_test,  bm_conf_test  = _run_classifier(
        _clf_best_match, sorted_primary, X_test_sel,
        {"min_weight_threshold": mwt, "partial_match_mode": pmm})
    bm_train, _  = _run_classifier(
        _clf_best_match, sorted_primary, X_train_sel,
        {"min_weight_threshold": mwt, "partial_match_mode": pmm})

    # Classifiers 2/3/4: use voting rule set
    sv_test,  sv_conf_test  = _run_classifier(
        _clf_soft_vote, sorted_voting, X_test_sel,
        {"sv_disease_threshold": sv_disease_threshold,
         "min_weight_threshold": mwt})
    sv_train, _  = _run_classifier(
        _clf_soft_vote, sorted_voting, X_train_sel,
        {"sv_disease_threshold": sv_disease_threshold,
         "min_weight_threshold": mwt})
    ah_test,  ah_conf_test  = _run_classifier(
        _clf_assume_healthy, sorted_voting, X_test_sel)
    ah_train, _  = _run_classifier(
        _clf_assume_healthy, sorted_voting, X_train_sel)
    as_test,  as_conf_test  = _run_classifier(
        _clf_assume_sick, sorted_voting, X_test_sel)
    as_train, _  = _run_classifier(
        _clf_assume_sick, sorted_voting, X_train_sel)

    # ── Overfitting gaps ──────────────────────────────────────────────────
    bm_gap = accuracy_score(y_train, bm_train) - accuracy_score(y_test, bm_test)
    sv_gap = accuracy_score(y_train, sv_train) - accuracy_score(y_test, sv_test)
    ah_gap = accuracy_score(y_train, ah_train) - accuracy_score(y_test, ah_test)
    as_gap = accuracy_score(y_train, as_train) - accuracy_score(y_test, as_test)

    # ── TEST PERFORMANCE TABLE ────────────────────────────────────────────
    print("TEST PERFORMANCE  "
          "(Acc=Accuracy  Bal=BalancedAcc  F1  Prec=Precision  "
          "Rec=Recall  AUC  TP FP FN TN)")
    print("-" * 90)
    _print_classifier_metrics(
        "1. Best-match",      y_test, bm_test, bm_conf_test)
    _print_classifier_metrics(
        "2. Soft voting",     y_test, sv_test, sv_conf_test)
    _print_classifier_metrics(
        "3. Assume healthy",  y_test, ah_test, ah_conf_test)
    _print_classifier_metrics(
        "4. Assume sick",     y_test, as_test, as_conf_test)
    print()

    # ── OVERFITTING TABLE ─────────────────────────────────────────────────
    print("OVERFITTING GAP  (Train Acc - Test Acc, lower is better)")
    print("-" * 90)
    for name, gap in [("1. Best-match",     bm_gap),
                       ("2. Soft voting",    sv_gap),
                       ("3. Assume healthy", ah_gap),
                       ("4. Assume sick",    as_gap)]:
        bar = "<-- !" if abs(gap) > 0.10 else ""
        print(f"  {name:<28} {gap:+.4f}  {bar}")
    print()

    # ── CONFUSION MATRICES ────────────────────────────────────────────────
    print("CONFUSION MATRICES  (rows=True, cols=Predicted)")
    print("-" * 90)
    for name, preds in [
        ("1. Best-match",    bm_test),
        ("2. Soft voting",   sv_test),
        ("3. Assume healthy",ah_test),
        ("4. Assume sick",   as_test),
    ]:
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        acc = accuracy_score(y_test, preds)
        print(f"  {name}")
        print(f"    Pred ->          0 (healthy)   1 (disease)")
        print(f"    True 0 (healthy)   {tn:6d}       {fp:6d}"
              f"   <- FP: healthy told sick")
        print(f"    True 1 (disease)   {fn:6d}       {tp:6d}"
              f"   <- FN: sick missed")
        print(f"    Correct: {tn+tp}/{len(y_test)} = {acc:.1%}")
        print()

    # ── PER-INSTANCE BREAKDOWN ────────────────────────────────────────────
    n_show = min(50, len(y_test))
    print(f"PER-INSTANCE BREAKDOWN  (first {n_show} test instances)")
    print("-" * 90)
    print(f"  {'i':>3}  True  "
          f"{'BestMatch':>9}  {'SoftVote':>8}  "
          f"{'AssumeH':>7}  {'AssumeS':>7}")
    print(f"  {'':>3}  {'':>4}  "
          f"{'Pred Conf':>9}  {'Pred Conf':>8}  "
          f"{'Pred Conf':>9}  {'Pred Conf':>9}")
    print("  " + "-" * 60)
    for i in range(n_show):
        true = y_test[i]
        def fmt(pred, conf, true):
            mark = " " if pred == true else "X"
            return f"{pred} {conf:.2f}{mark}"
        print(f"  {i:>3}  {true:>4}  "
              f"{fmt(bm_test[i], bm_conf_test[i], true):>9}  "
              f"{fmt(sv_test[i], sv_conf_test[i], true):>8}  "
              f"{fmt(ah_test[i], ah_conf_test[i], true):>9}  "
              f"{fmt(as_test[i], as_conf_test[i], true):>9}")


def evaluate(miner, X_train_sel, y_train, X_test_sel, y_test,
             model_name, selector_name, miner_name, purity, threshold,
             min_weight_threshold=0.05):
    print("="*90)
    print("                       LAD EVALUATION")
    print("="*90)
    print("PIPELINE SETTINGS")
    print(f"  Selector              : {selector_name}")
    print(f"  Pattern Miner         : {miner_name}")
    print(f"  Purity threshold      : {purity}")
    print(f"  Support threshold     : {threshold}")
    print(f"  Min weight threshold  : {min_weight_threshold}")
    if miner_name == "eager":
        print(f"  Min rule support      : {getattr(miner, 'min_rule_support', '-')}")
    if miner_name == "maxpatterns":
        muc = getattr(miner, 'min_unique_coverage', 0)
        if muc > 0:
            print(f"  Min unique coverage   : {muc}")
    print()

    if not miner.rules:
        print("[Error] No rules available. Cannot evaluate.")
        return

    n_rules       = len(miner.rules)
    n_cutpoints   = len(miner.binarizer.cutpoints)
    n_features    = X_train_sel.shape[1]
    model_size_kb = len(pickle.dumps({
        "cutpoints": miner.binarizer.cutpoints,
        "selected":  miner.selector.best_subset,
        "rules":     miner.rules,
    })) / 1024

    start = time.time()
    test_pred, test_exact, test_conf = predict_all(
        miner, X_test_sel, min_weight_threshold, PARTIAL_MATCH_MODE
    )
    pred_time_ms = (time.time() - start) / len(X_test_sel) * 1000
    train_pred, _, _ = predict_all(
        miner, X_train_sel, min_weight_threshold, PARTIAL_MATCH_MODE)

    train_acc       = accuracy_score(y_train, train_pred)
    test_acc        = accuracy_score(y_test,  test_pred)
    overfitting_gap = train_acc - test_acc
    bal_acc         = balanced_accuracy_score(y_test, test_pred)
    f1              = f1_score(y_test, test_pred)
    prec            = precision_score(y_test, test_pred, zero_division=0)
    rec             = recall_score(y_test, test_pred, zero_division=0)
    auc             = roc_auc_score(y_test, test_conf)
    exact_cov       = test_exact.mean()
    tn, fp, fn, tp  = confusion_matrix(y_test, test_pred).ravel()

    print(f"Model                 : {model_name}")
    print(f"Rules                 : {n_rules}")
    print(f"Cutpoints             : {n_cutpoints}")
    print(f"Selected Features     : {n_features}")
    print(f"Model Size            : {model_size_kb:.1f} KB")
    print(f"Prediction Speed      : {pred_time_ms:.2f} ms per sample")
    print(f"Exact Rule Coverage   : {exact_cov:.1%} "
          f"({int(test_exact.sum())}/{len(y_test)} fully explained)")
    print()
    print(f"Train Accuracy        : {train_acc:.4f}")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print(f"OVERFITTING GAP       : {overfitting_gap:+.4f} "
          f"<-- <-- <-- <-- <-- <-- <--")
    print()
    print("TEST PERFORMANCE")
    print(f"  Accuracy            : {test_acc:.4f}")
    print(f"  Balanced Accuracy   : {bal_acc:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print(f"  ROC-AUC (ranking)   : {auc:.4f}"
          f"  <- overlap scores, not calibrated probabilities")
    print()
    print("FULL CONFUSION MATRIX")
    print("                        Predicted ->")
    print("                          No Disease (0)    Disease (1)")
    print(f"True No Disease (0) ->      {tn:4d}             {fp:4d}"
          f"    <- FP: Healthy wrongly told sick")
    print(f"True Disease    (1) ->      {fn:4d}             {tp:4d}"
          f"    <- FN: Sick patient missed")
    print()
    print(f"-> {fp} healthy patients incorrectly predicted as having disease")
    print(f"-> {fn} sick patients incorrectly predicted as healthy "
          f"(False Negative <- more dangerous)")
    print(f"Correct predictions   : {tn + tp}/{len(y_test)} = {test_acc:.1%}")


def evaluate_lazy(lazy, X_train_sel, y_train, X_test_sel, y_test,
                  selector_name, purity, min_support):
    print("="*90)
    print("                 LAD EVALUATION  --  LAZY PATTERNS")
    print("="*90)
    print("PIPELINE SETTINGS")
    print(f"  Selector          : {selector_name}")
    print(f"  Pattern Miner     : lazy")
    print(f"  Purity threshold  : {purity}")
    print(f"  Min support       : {min_support}")
    print()

    def run_lazy(X):
        results  = lazy.predict(X)
        labels   = np.array([r["label"]       for r in results])
        purities = np.array([r["purity"]      for r in results])
        supports = np.array([r["support"]     for r in results])
        exacts   = np.array([r["exact_match"] for r in results])
        stages   = np.array([r["stage"]       for r in results])
        return labels, purities, supports, exacts, stages, results

    start = time.time()
    test_pred, test_pur, test_sup, test_exact, test_stages, test_results = \
        run_lazy(X_test_sel)
    pred_time_ms = (time.time() - start) / len(X_test_sel) * 1000
    train_pred, _, _, _, _, _ = run_lazy(X_train_sel)

    n_test     = len(y_test)
    n_exact    = int(np.sum(test_stages == "exact"))
    n_pruned   = int(np.sum(test_stages == "pruned"))
    n_fallback = int(np.sum(test_stages == "fallback"))

    acc_exact    = (accuracy_score(y_test[test_stages == "exact"],
                                   test_pred[test_stages == "exact"])
                    if n_exact    > 0 else float("nan"))
    acc_pruned   = (accuracy_score(y_test[test_stages == "pruned"],
                                   test_pred[test_stages == "pruned"])
                    if n_pruned   > 0 else float("nan"))
    acc_fallback = (accuracy_score(y_test[test_stages == "fallback"],
                                   test_pred[test_stages == "fallback"])
                    if n_fallback > 0 else float("nan"))

    train_acc       = accuracy_score(y_train, train_pred)
    test_acc        = accuracy_score(y_test,  test_pred)
    overfitting_gap = train_acc - test_acc
    bal_acc         = balanced_accuracy_score(y_test, test_pred)
    f1              = f1_score(y_test, test_pred)
    prec            = precision_score(y_test, test_pred, zero_division=0)
    rec             = recall_score(y_test, test_pred, zero_division=0)
    tn, fp, fn, tp  = confusion_matrix(y_test, test_pred).ravel()

    print("STAGE DIAGNOSTICS")
    for stage, n, acc in [("exact",    n_exact,    acc_exact),
                           ("pruned",   n_pruned,   acc_pruned),
                           ("fallback", n_fallback, acc_fallback)]:
        acc_str = f"accuracy={acc:.4f}" if not np.isnan(acc) else ""
        print(f"  Stage {stage:8s} : {n:4d}/{n_test} "
              f"({n/n_test:.1%})  {acc_str}")
    print()

    if n_fallback / n_test > 0.3:
        print(f"  [DiagnosticHint] {n_fallback/n_test:.1%} fallback rate is high.")
        print(f"    -- Lower LAZY_PURITY (currently {purity}) -- try 0.65 or 0.60")
        print(f"    -- Lower LAZY_MIN_SUPPORT (currently {min_support}) -- try 2")
        print(f"    -- Use a binarizer mode with more features")
        print()

    print(f"Prediction Speed      : {pred_time_ms:.2f} ms per sample")
    print(f"Avg Rule Purity       : {test_pur.mean():.4f}")
    print(f"Avg Rule Support      : {test_sup.mean():.1f}")
    print()
    print(f"Train Accuracy        : {train_acc:.4f}")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print(f"OVERFITTING GAP       : {overfitting_gap:+.4f} "
          f"<-- <-- <-- <-- <-- <-- <--")
    print()
    print("TEST PERFORMANCE")
    print(f"  Accuracy            : {test_acc:.4f}")
    print(f"  Balanced Accuracy   : {bal_acc:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print()
    print("FULL CONFUSION MATRIX")
    print("                        Predicted ->")
    print("                          No Disease (0)    Disease (1)")
    print(f"True No Disease (0) ->      {tn:4d}             {fp:4d}"
          f"    <- FP: Healthy wrongly told sick")
    print(f"True Disease    (1) ->      {fn:4d}             {tp:4d}"
          f"    <- FN: Sick patient missed")
    print()
    print(f"-> {fp} healthy patients incorrectly predicted as having disease")
    print(f"-> {fn} sick patients incorrectly predicted as healthy "
          f"(False Negative <- more dangerous)")
    print(f"Correct predictions   : {tn + tp}/{len(y_test)} = {test_acc:.1%}")

    print(f"\nFirst {min(100, n_test)} lazy predictions:")
    for i in range(min(100, n_test)):
        r        = test_results[i]
        rule_str = " AND ".join(r["rule"])[:80]
        print(f"  True={y_test[i]} | Pred={r['label']} | "
              f"Stage={r['stage']:8s} | Purity={r['purity']:.2f} | "
              f"Support={r['support']:4d} | Rule: {rule_str}...")


def cross_validate_pipeline(X, y, feature_names,
                             binarizer_type,
                             bf_mode, bf_top_k, bf_min_support,
                             bf_min_interval_frac, bf_categorical_cols,
                             bf_max_group_size,
                             dt_mode, dt_max_depth, dt_min_samples,
                             selector_name, h_weight,
                             miner_name, purity, threshold,
                             ga_generations, ga_pop_size, ga_sharing_sigma,
                             min_weight_threshold,
                             n_splits=5):
    """
    Full stratified k-fold CV of the entire pipeline including binarization
    and feature selection on each fold. Reports mean +/- std accuracy and F1.
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION  ({n_splits}-fold stratified)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # Binarize
        _, Xbin_tr, Xbin_va, bin_names = binarize(
            Xtr, ytr, Xva, feature_names,
            binarizer_type       = binarizer_type,
            bf_mode              = bf_mode,
            bf_top_k             = bf_top_k,
            bf_min_support       = bf_min_support,
            bf_min_interval_frac = bf_min_interval_frac,
            bf_categorical_cols  = bf_categorical_cols,
            bf_max_group_size    = bf_max_group_size,
            dt_mode              = dt_mode,
            dt_max_depth         = dt_max_depth,
            dt_min_samples       = dt_min_samples,
        )

        # Select features (silence verbose output during CV)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sel, Xsel_tr, y_clean = select_features(
                selector_name, Xbin_tr, ytr, bin_names, h_weight=h_weight
            )
        Xsel_va = Xbin_va[:, sel.best_subset]

        # Mine and predict
        miner = _make_miner(
            miner_name, _, sel, purity, threshold,
            ga_generations, ga_pop_size, ga_sharing_sigma, verbose=False
        )
        # Need binarizer in the miner — re-fetch from binarize
        _, Xbin_tr2, Xbin_va2, bin_names2 = binarize(
            Xtr, ytr, Xva, feature_names,
            binarizer_type=binarizer_type,
            bf_mode=bf_mode, bf_top_k=bf_top_k,
            bf_min_support=bf_min_support,
            bf_min_interval_frac=bf_min_interval_frac,
            bf_categorical_cols=bf_categorical_cols,
            bf_max_group_size=bf_max_group_size,
            dt_mode=dt_mode, dt_max_depth=dt_max_depth,
            dt_min_samples=dt_min_samples,
        )
        with contextlib.redirect_stdout(buf):
            sel2, Xsel_tr2, y_clean2 = select_features(
                selector_name, Xbin_tr2, ytr, bin_names2, h_weight=h_weight
            )
        miner2 = _make_miner(
            miner_name, sel2.binarizer if hasattr(sel2, "binarizer") else _,
            sel2, purity, threshold,
            ga_generations, ga_pop_size, ga_sharing_sigma, verbose=False
        )

        with contextlib.redirect_stdout(buf):
            miner2.fit(Xsel_tr2, y_clean2, original_feature_names=feature_names)

        if not miner2.rules:
            print(f"  Fold {fold}: no rules generated, skipping.")
            continue

        Xsel_va2 = Xbin_va2[:, sel2.best_subset]
        preds, _, _ = predict_all(miner2, Xsel_va2, min_weight_threshold)
        fold_accs.append(accuracy_score(yva, preds))
        fold_f1s.append(f1_score(yva, preds, zero_division=0))
        print(f"  Fold {fold}: acc={fold_accs[-1]:.4f}  "
              f"f1={fold_f1s[-1]:.4f}")

    if fold_accs:
        print(f"\n  CV Accuracy : {np.mean(fold_accs):.4f} "
              f"+/- {np.std(fold_accs):.4f}")
        print(f"  CV F1       : {np.mean(fold_f1s):.4f} "
              f"+/- {np.std(fold_f1s):.4f}")
    else:
        print("  No folds completed successfully.")

    return fold_accs, fold_f1s


# ================================================================
# ENTRY POINT
# ================================================================

def run_pipeline():
    # Step 1 -- Load and split
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(
        DATA_PATH, TEST_SIZE, RANDOM_STATE
    )

    # Step 2 -- Binarize
    binarizer, Xbin_train, Xbin_test, bin_feature_names = binarize(
        X_train, y_train, X_test, feature_names,
        binarizer_type       = BINARIZER_TYPE,
        bf_mode              = BF_MODE,
        bf_top_k             = BF_TOP_K,
        bf_min_support       = BF_MIN_SUPPORT,
        bf_min_interval_frac = BF_MIN_INTERVAL_FRAC,
        bf_categorical_cols  = BF_CATEGORICAL_COLS,
        bf_max_group_size    = BF_MAX_GROUP_SIZE,
        dt_mode              = DT_MODE,
        dt_max_depth         = DT_MAX_DEPTH,
        dt_min_samples       = DT_MIN_SAMPLES,
    )

    # Save originals before select_features() overwrites y_train with y_clean
    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()

    # Step 3 -- Optionally tune A* h_weight, then select features
    h_weight = ASTAR_H_WEIGHT
    if TUNE_ASTAR and SELECTOR in ("astar", "mutualinfo_astar"):
        h_weight = tune_astar_weight(
            Xbin_train, y_train, bin_feature_names,
            ASTAR_H_WEIGHT_CANDIDATES, ASTAR_TUNE_EXPANSIONS
        )

    selector, Xsel_train, y_train = select_features(
        SELECTOR, Xbin_train, y_train, bin_feature_names,
        h_weight=h_weight,
        class_weight_disease=CLASS_WEIGHT_DISEASE,
    )

    Xsel_test = Xbin_test[:, selector.best_subset]
    print(f"Test matrix after selection : {Xsel_test.shape}\n")

    # Step 4 + 5 -- Pattern mining and evaluation
    if PATTERN_MINER == "lazy":
        lazy = LazyPatterns(
            binarizer    = binarizer,
            selector     = selector,
            purity       = LAZY_PURITY,
            min_support  = LAZY_MIN_SUPPORT,
            verbose      = True,
        )
        lazy.fit(Xsel_train, y_train, original_feature_names=feature_names)

        evaluate_lazy(
            lazy          = lazy,
            X_train_sel   = Xsel_train,
            y_train       = y_train,
            X_test_sel    = Xsel_test,
            y_test        = y_test,
            selector_name = SELECTOR,
            purity        = LAZY_PURITY,
            min_support   = LAZY_MIN_SUPPORT,
        )

    else:
        # Optionally tune purity before mining
        purity = PURITY
        if TUNE_PURITY:
            purity = tune_purity(
                Xsel_train, y_train, binarizer, selector,
                feature_names, PATTERN_MINER, THRESHOLD,
                PURITY_CANDIDATES,
                GA_GENERATIONS, GA_POP_SIZE, GA_SHARING_SIGMA
            )

        if PATTERN_MINER in ("hybrid_mp_eager", "hybrid_eager_mp"):
            miner = _make_hybrid_miner(
                PATTERN_MINER, binarizer, selector,
                purity, THRESHOLD, feature_names,
            )
            miner.fit(Xsel_train, y_train,
                      original_feature_names=feature_names)
            if miner.rules:
                print(f"\nTOP {len(miner.rules)} HYBRID RULES DISCOVERED:")
                miner.print_rules(top_n=len(miner.rules))
            else:
                print("\n[Warning] HybridMiner produced no rules.")
        else:
            miner = mine_patterns(
                PATTERN_MINER, binarizer, selector, Xsel_train, y_train,
                feature_names, purity, THRESHOLD,
                GA_GENERATIONS, GA_POP_SIZE, GA_SHARING_SIGMA,
            )

        print_predictions(miner, Xsel_test, y_test, MIN_WEIGHT_THRESHOLD)

        evaluate(
            miner                = miner,
            X_train_sel          = Xsel_train,
            y_train              = y_train,
            X_test_sel           = Xsel_test,
            y_test               = y_test,
            model_name           = "LAD Heart Disease",
            selector_name        = SELECTOR,
            miner_name           = PATTERN_MINER,
            purity               = purity,
            threshold            = THRESHOLD,
            min_weight_threshold = MIN_WEIGHT_THRESHOLD,
        )

        # Build overlapping rule set for voting classifiers if enabled
        overlapping_rules = None
        if USE_OVERLAPPING_RULES_FOR_VOTING:
            print("\nBuilding overlapping rule set for voting classifiers...")
            overlapping_rules = _build_overlapping_rules(
                Xsel_train, y_train, binarizer, selector,
                purity, THRESHOLD, OVERLAPPING_MIN_COV, feature_names
            )
            print(f"  Overlapping rules built: {len(overlapping_rules)}")

        evaluate_all_classifiers(
            miner                = miner,
            X_train_sel          = Xsel_train,
            y_train              = y_train,
            X_test_sel           = Xsel_test,
            y_test               = y_test,
            model_name           = "LAD Heart Disease",
            selector_name        = SELECTOR,
            miner_name           = PATTERN_MINER,
            purity               = purity,
            threshold            = THRESHOLD,
            min_weight_threshold = MIN_WEIGHT_THRESHOLD,
            sv_disease_threshold = SV_DISEASE_THRESHOLD,
            overlapping_rules    = overlapping_rules,
        )

        # Optional cross-validation — uses original unsplit train data
        if RUN_CV:
            cross_validate_pipeline(
                X_train_orig, y_train_orig, feature_names,
                binarizer_type       = BINARIZER_TYPE,
                bf_mode              = BF_MODE,
                bf_top_k             = BF_TOP_K,
                bf_min_support       = BF_MIN_SUPPORT,
                bf_min_interval_frac = BF_MIN_INTERVAL_FRAC,
                bf_categorical_cols  = BF_CATEGORICAL_COLS,
                bf_max_group_size    = BF_MAX_GROUP_SIZE,
                dt_mode              = DT_MODE,
                dt_max_depth         = DT_MAX_DEPTH,
                dt_min_samples       = DT_MIN_SAMPLES,
                selector_name        = SELECTOR,
                h_weight             = h_weight,
                miner_name           = PATTERN_MINER,
                purity               = purity,
                threshold            = THRESHOLD,
                ga_generations       = GA_GENERATIONS,
                ga_pop_size          = GA_POP_SIZE,
                ga_sharing_sigma     = GA_SHARING_SIGMA,
                min_weight_threshold = MIN_WEIGHT_THRESHOLD,
                n_splits             = CV_SPLITS,
            )


if __name__ == "__main__":
    run_pipeline()