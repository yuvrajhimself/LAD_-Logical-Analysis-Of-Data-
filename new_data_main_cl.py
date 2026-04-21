"""
LAD Heart Disease Pipeline
==========================
Configure your choices below, then run. All pipeline logic lives in functions.
"""

# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH    = "Z_Alizadeh_sani.csv"
TEST_SIZE    = 0.25
RANDOM_STATE = 42

# ── Binarizer ────────────────────────────────────────────────────────────────
# "bruteforce" | "decisiontree"
BINARIZER_TYPE = "decisiontree"

# BruteForceBinarizer  (BINARIZER_TYPE = "bruteforce")
BF_MODE              = "greedy"
BF_TOP_K             = 20
BF_MIN_SUPPORT       = 5
BF_MIN_INTERVAL_FRAC = 0.05
# Cleveland categorical column indices (0-based):
#   1=cp, 2=fbs, 5=restecg, 6=exang, 8=slope, 10=ca, 11=thal
BF_CATEGORICAL_COLS  = [1, 2, 5, 6, 8, 10, 11]

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
ASTAR_TUNE_EXPANSIONS     = 5000   # expansions budget per candidate during tuning

# ── Pattern Miner ────────────────────────────────────────────────────────────
# "maxpatterns" | "eager" | "genetic" | "lazy"
PATTERN_MINER = "eager"

# Purity for eager miners (maxpatterns / eager / genetic).
# Set TUNE_PURITY = True to auto-select from PURITY_CANDIDATES via 5-fold CV.
PURITY                = 0.95
TUNE_PURITY           = False
PURITY_CANDIDATES     = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

THRESHOLD = 0

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
from BruteForceBinarizer_cl import BruteForceBinarizer
from DecisionTreeCutpointBinarizerV2 import DecisionTreeCutpointBinarizerV2
from LADScorer_cl import lad_score

# ================================================================
# PIPELINE FUNCTIONS
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path, test_size, random_state):
    # 1. Load the dataset
    df = pd.read_csv(file_path)
    
    # 2. Fix the specific 'Fmale' typo and strip whitespace
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.replace({'Fmale': 'Female'})
    
    # 3. Explicitly map the Target (Cath)
    # The last column in Alizadeh Sani is usually 'Cath'
    df.iloc[:, -1] = df.iloc[:, -1].map({'Cad': 1, 'Normal': 0})
    
    # 4. Convert all categorical features to numeric
    # This checks every column; if it's text, it converts it to a code
    for col in df.columns[:-1]:
        if df[col].dtype == 'object':
            # Map binary strings if they exist
            if set(df[col].unique()) <= {'Yes', 'No'}:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
            elif set(df[col].unique()) <= {'Male', 'Female'}:
                df[col] = df[col].map({'Male': 1, 'Female': 0})
            else:
                # For columns like VHD or Region RWMA
                df[col] = pd.factorize(df[col])[0]
                
    # 5. Final Safety Check: Force everything to float
    # This ensures that X contains NO strings before going to the binarizer
    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    
    # If any NaNs were created by the numeric conversion, fill them with 0
    X_df = X_df.fillna(0)
    
    X = X_df.values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    feature_names = X_df.columns.tolist()

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Data cleaning complete. No strings remaining in features.")
    return X_train, X_test, y_train, y_test, feature_names

def binarize(X_train, y_train, X_test, feature_names,
             binarizer_type,
             bf_mode, bf_top_k, bf_min_support,
             bf_min_interval_frac, bf_categorical_cols,
             dt_mode, dt_max_depth, dt_min_samples):
    if binarizer_type == "bruteforce":
        binarizer = BruteForceBinarizer(
            mode=bf_mode,
            top_k_per_feature=bf_top_k,
            min_support=bf_min_support,
            min_interval_fraction=bf_min_interval_frac,
            categorical_cols=bf_categorical_cols,
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
                    h_weight=5):
    if selector_name == "greedy":
        selector = GreedyLADSelector()
        selector.fit(Xbin_train, y_train, bin_feature_names)
    elif selector_name == "astar":
        selector = AStarFeatureSelector(h_weight=h_weight)
        selector.fit(Xbin_train, y_train, bin_feature_names)
    elif selector_name == "mutualinfo":
        selector = MutualInfoGreedySelector()
        selector.fit(Xbin_train, y_train)
    elif selector_name == "mutualinfo_astar":
        selector = MutualInfoAStarSelector(h_weight=h_weight)
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
                ga_generations, ga_pop_size, ga_sharing_sigma, verbose=True):
    if miner_name == "maxpatterns":
        return MaxPatterns(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=verbose, threshold=threshold,
        )
    elif miner_name == "eager":
        return Eager(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=verbose, threshold=threshold,
        )
    elif miner_name == "genetic":
        return GeneticRuleMiner(
            binarizer=binarizer, selector=selector,
            purity=purity, verbose=verbose, threshold=threshold,
            n_generations=ga_generations, pop_size=ga_pop_size,
            sharing_sigma=ga_sharing_sigma,
        )
    else:
        raise ValueError(
            f"Unknown miner: '{miner_name}'. "
            f"Choose from: maxpatterns, eager, genetic"
        )


def tune_purity(Xsel_train, y_train, binarizer, selector,
                feature_names, miner_name, threshold,
                candidates, ga_generations, ga_pop_size, ga_sharing_sigma,
                n_splits=5):
    """
    5-fold stratified CV over purity candidates.
    Returns the candidate with the highest mean CV accuracy.
    """
    print("\n=== Tuning purity threshold ===")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    best_purity = candidates[0]
    best_score  = -1.0

    for purity in candidates:
        fold_scores = []
        for tr_idx, va_idx in skf.split(Xsel_train, y_train):
            Xtr, ytr = Xsel_train[tr_idx], y_train[tr_idx]
            Xva, yva = Xsel_train[va_idx], y_train[va_idx]

            miner = _make_miner(
                miner_name, binarizer, selector, purity, threshold,
                ga_generations, ga_pop_size, ga_sharing_sigma, verbose=False
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
        ga_generations, ga_pop_size, ga_sharing_sigma, verbose=True
    )
    miner.fit(Xsel_train, y_train, original_feature_names=feature_names)

    if miner.rules:
        print(f"\nTOP {len(miner.rules)} RULES DISCOVERED ON TRAINING DATA:")
        miner.print_rules(top_n=len(miner.rules))
    else:
        print("\n[Warning] No rules were generated. "
              "Try lowering PURITY or THRESHOLD in the config.")
    return miner


# ================================================================
# PREDICTION
# ================================================================

def _predict_row(row, sorted_rules, min_weight_threshold=0.05):
    """
    Pass 1 — exact match: return immediately on first full match.
    Pass 2 — partial match: among rules above min_weight_threshold,
             pick the one with highest overlap fraction; break ties
             by weight. This separates overlap quality from rule
             dominance rather than multiplying them together.
    Pass 3 — fallback: if no rule clears the weight threshold,
             use the highest-weight rule regardless.
    Returns (label, is_exact, overlap_score).
    """
    # Pass 1
    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return rule["label"], True, 1.0

    # Pass 2
    best_label, best_overlap, best_weight = None, -1.0, -1.0
    for rule in sorted_rules:
        if rule["weight"] < min_weight_threshold:
            continue
        attrs, values = rule["attrs"], rule["values"]
        overlap = sum(
            row[attrs[i]] == values[i] for i in range(len(attrs))
        ) / len(attrs)
        if (overlap > best_overlap or
                (overlap == best_overlap and rule["weight"] > best_weight)):
            best_overlap = overlap
            best_weight  = rule["weight"]
            best_label   = rule["label"]

    if best_label is not None:
        return best_label, False, best_overlap

    # Pass 3 — no rule cleared the weight threshold
    if sorted_rules:
        return sorted_rules[0]["label"], False, 0.0

    return None, False, 0.0


def predict_all(miner, X_selected, min_weight_threshold=0.05):
    if not miner.rules:
        raise RuntimeError("Miner has no rules. Cannot predict.")
    sorted_rules = sorted(miner.rules, key=lambda r: r["weight"], reverse=True)
    labels, exacts, scores = [], [], []
    for row in X_selected:
        label, exact, score = _predict_row(
            row, sorted_rules, min_weight_threshold
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
            row, sorted_rules, min_weight_threshold
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


# ================================================================
# EVALUATION
# ================================================================

def evaluate(miner, X_train_sel, y_train, X_test_sel, y_test,
             model_name, selector_name, miner_name, purity, threshold,
             min_weight_threshold=0.05):
    print("="*90)
    print("                       LAD EVALUATION")
    print("="*90)
    print("PIPELINE SETTINGS")
    print(f"  Selector              : {selector_name}")
    print(f"  Pattern Miner         : {miner_name}")
    print(f"  Binarizer             : {BINARIZER_TYPE}")
    print(f"  Purity threshold      : {purity}")
    print(f"  Support threshold     : {threshold}")
    print(f"  Min weight threshold  : {min_weight_threshold}")
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
        miner, X_test_sel, min_weight_threshold
    )
    pred_time_ms = (time.time() - start) / len(X_test_sel) * 1000
    train_pred, _, _ = predict_all(miner, X_train_sel, min_weight_threshold)

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
        h_weight=h_weight
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