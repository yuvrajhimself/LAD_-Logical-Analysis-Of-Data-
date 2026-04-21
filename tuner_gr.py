"""
LAD Pipeline Staged Hyperparameter Tuner
=========================================
Searches parameters in three nested stages to avoid redundant computation:

  Stage 1 -- Binarizer config
    Stage 2 -- Selector config  (binarized data reused across all selector trials)
      Stage 3 -- Miner config   (selected data reused across all miner trials)

For N_bin binarizer configs x N_sel selector configs x N_min miner configs:
  - Binarizer runs  : N_bin
  - Selector runs   : N_bin * N_sel
  - Miner/eval runs : N_bin * N_sel * N_min  (cheapest step)

All results are logged to SQLite immediately after each miner trial.
If the tuner crashes or is stopped, re-running it skips completed trials
(resume is per miner-trial, so no work is ever lost).

Usage:
    python tuner.py

Output:
    lad_tuner.db  -- SQLite database, one row per miner trial
    Console       -- live progress + leaderboard at the end
"""

import sys, os, time, json, hashlib, sqlite3, gc
import traceback, io, contextlib
import numpy as np
import pandas as pd
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

from BruteForceBinarizer_cl      import BruteForceBinarizer
from AStarFeatureSelector_cl     import AStarFeatureSelector
from GreedyLADSelector_cl        import GreedyLADSelector
from MutualInfoGreedySelector_cl import MutualInfoGreedySelector
from MutualInfoAStarSelector_cl  import MutualInfoAStarSelector
from MaxPatterns_cl              import MaxPatterns
from Eager_cl                    import Eager

# ================================================================
# SETTINGS
# ================================================================

DATA_PATH        = "Heart_disease_cleveland_new.csv"
TEST_SIZE        = 0.25
RANDOM_STATE     = 42
DB_PATH          = "lad_tuner.db"
CATEGORICAL_COLS = [1, 2, 5, 6, 8, 10, 11]

# A* expansion cap for tuner — keeps each selector run bounded.
# 1110999 (default) can take minutes on large binary spaces.
# 50000 is enough to find a good subset without hanging.
ASTAR_MAX_EXPANSIONS = 10000

# Maximum seconds allowed for a single miner trial (fit + evaluate).
# Trials that exceed this are logged as status='timeout' and skipped.
# Eager with many features can run indefinitely — this caps it.
TRIAL_TIMEOUT_SECONDS = 600   # 10 minutes

# ================================================================
# PARAMETER GRIDS
# ================================================================

BINARIZER_GRID = [
    # bf_interval_frac controls minimum gap between two numeric cutpoints
    # (as a fraction of the feature range). Smaller = denser cutpoints.
    {"bf_top_k": 20, "bf_min_support": 3, "bf_max_group_size": 5, "bf_interval_frac": 0.05},
    {"bf_top_k": 20, "bf_min_support": 3, "bf_max_group_size": 5, "bf_interval_frac": 0.04},
    {"bf_top_k": 20, "bf_min_support": 3, "bf_max_group_size": 5, "bf_interval_frac": 0.03},
    {"bf_top_k": 20, "bf_min_support": 3, "bf_max_group_size": 5, "bf_interval_frac": 0.06},

    {"bf_top_k": 20, "bf_min_support": 5, "bf_max_group_size": 5, "bf_interval_frac": 0.05},
    {"bf_top_k": 20, "bf_min_support": 5, "bf_max_group_size": 5, "bf_interval_frac": 0.04},
    {"bf_top_k": 20, "bf_min_support": 5, "bf_max_group_size": 5, "bf_interval_frac": 0.03},
    {"bf_top_k": 20, "bf_min_support": 5, "bf_max_group_size": 5, "bf_interval_frac": 0.06},
]

# h_weight only applies to astar / mutualinfo_astar; stored as 0 for others
SELECTOR_GRID = [
    {"selector": "astar",            "h_weight": 0},
    {"selector": "astar",            "h_weight": 1},
    {"selector": "astar",            "h_weight": 2},
    {"selector": "astar",            "h_weight": 3},
    {"selector": "astar",            "h_weight": 4},
    {"selector": "astar",            "h_weight": 5},
    {"selector": "astar",            "h_weight": 6},
    {"selector": "astar",            "h_weight": 7},
    {"selector": "astar",            "h_weight": 8},
    {"selector": "astar",            "h_weight": 9},
    {"selector": "astar",            "h_weight": 10},
    {"selector": "mutualinfo_astar", "h_weight": 0},
    {"selector": "mutualinfo_astar", "h_weight": 1},
    {"selector": "mutualinfo_astar", "h_weight": 2},
    {"selector": "mutualinfo_astar", "h_weight": 3},
    {"selector": "mutualinfo_astar", "h_weight": 4},
    {"selector": "mutualinfo_astar", "h_weight": 5},
    {"selector": "mutualinfo_astar", "h_weight": 6},
    {"selector": "mutualinfo_astar", "h_weight": 7},
    {"selector": "mutualinfo_astar", "h_weight": 8},
    {"selector": "mutualinfo_astar", "h_weight": 9},
    {"selector": "mutualinfo_astar", "h_weight": 10},
]

# mp_min_unique_cov only applies to maxpatterns
# eager_min_support only applies to eager
MINER_GRID = [
    # MaxPatterns
    {"miner": "maxpatterns", "purity": 0.75, "mp_min_unique_cov": 0, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.75, "mp_min_unique_cov": 3, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.80, "mp_min_unique_cov": 0, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.80, "mp_min_unique_cov": 3, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.85, "mp_min_unique_cov": 0, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.85, "mp_min_unique_cov": 3, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.90, "mp_min_unique_cov": 0, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 0.90, "mp_min_unique_cov": 3, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 1.00, "mp_min_unique_cov": 0, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "maxpatterns", "purity": 1.00, "mp_min_unique_cov": 3, "eager_min_support": 0, "threshold": 3, "min_weight_thr": 0.05},
    # Eager
    {"miner": "eager", "purity": 0.75, "mp_min_unique_cov": 0, "eager_min_support": 3, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.75, "mp_min_unique_cov": 0, "eager_min_support": 5, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.80, "mp_min_unique_cov": 0, "eager_min_support": 3, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.80, "mp_min_unique_cov": 0, "eager_min_support": 5, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.85, "mp_min_unique_cov": 0, "eager_min_support": 3, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.85, "mp_min_unique_cov": 0, "eager_min_support": 5, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.90, "mp_min_unique_cov": 0, "eager_min_support": 3, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 0.90, "mp_min_unique_cov": 0, "eager_min_support": 5, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 1.00, "mp_min_unique_cov": 0, "eager_min_support": 3, "threshold": 3, "min_weight_thr": 0.05},
    {"miner": "eager", "purity": 1.00, "mp_min_unique_cov": 0, "eager_min_support": 5, "threshold": 3, "min_weight_thr": 0.05},
]

# Total = 16 binarizers x 8 selectors x 40 miners = 5120 miner trials
# Binarizer runs = 16, Selector runs = 128, Miner/eval runs = 5120

# ================================================================
# DATABASE
# ================================================================

def init_db(db_path):
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            trial_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            config_hash       TEXT UNIQUE,

            bf_top_k          INTEGER,
            bf_min_support    INTEGER,
            bf_max_group_size INTEGER,
            bf_interval_frac  REAL,
            n_cutpoints       INTEGER,

            selector          TEXT,
            h_weight          INTEGER,
            n_selected_feats  INTEGER,

            miner             TEXT,
            purity            REAL,
            threshold         INTEGER,
            mp_min_unique_cov INTEGER,
            eager_min_support INTEGER,
            min_weight_thr    REAL,
            n_rules           INTEGER,

            test_acc          REAL,
            test_bal_acc      REAL,
            test_f1           REAL,
            test_prec         REAL,
            test_rec          REAL,
            test_auc          REAL,
            train_acc         REAL,
            overfit_gap       REAL,
            tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER,

            sv_test_acc       REAL,
            sv_test_f1        REAL,
            sv_train_acc      REAL,

            ah_test_acc       REAL,
            ah_test_f1        REAL,

            as_test_acc       REAL,
            as_test_f1        REAL,

            runtime_s         REAL,
            status            TEXT,
            error_msg         TEXT,
            timestamp         TEXT
        )
    """)
    con.commit()
    return con


def make_hash(b_cfg, s_cfg, m_cfg):
    combined = {**b_cfg, **s_cfg, **m_cfg}
    return hashlib.md5(
        json.dumps(combined, sort_keys=True).encode()
    ).hexdigest()[:20]


def is_done(con, h):
    row = con.execute(
        "SELECT 1 FROM trials WHERE config_hash=?", (h,)
    ).fetchone()
    return row is not None


def log_trial(con, h, b_cfg, s_cfg, m_cfg, metrics, runtime, status, error=""):
    con.execute("""
        INSERT OR REPLACE INTO trials (
            config_hash,
            bf_top_k, bf_min_support, bf_max_group_size,
            bf_interval_frac, n_cutpoints,
            selector, h_weight, n_selected_feats,
            miner, purity, threshold, mp_min_unique_cov, eager_min_support,
            min_weight_thr, n_rules,
            test_acc, test_bal_acc, test_f1, test_prec, test_rec, test_auc,
            train_acc, overfit_gap, tp, fp, fn, tn,
            sv_test_acc, sv_test_f1, sv_train_acc,
            ah_test_acc, ah_test_f1,
            as_test_acc, as_test_f1,
            runtime_s, status, error_msg, timestamp
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,
            ?,?,?,datetime('now')
        )
    """, (
        h,
        b_cfg["bf_top_k"], b_cfg["bf_min_support"],
        b_cfg["bf_max_group_size"],
        b_cfg["bf_interval_frac"], metrics.get("n_cutpoints", 0),
        s_cfg["selector"], s_cfg["h_weight"], metrics.get("n_selected", 0),
        m_cfg["miner"], m_cfg["purity"],
        m_cfg["threshold"],
        m_cfg["mp_min_unique_cov"], m_cfg["eager_min_support"],
        m_cfg["min_weight_thr"], metrics.get("n_rules", 0),
        metrics.get("test_acc",  float("nan")),
        metrics.get("test_bal",  float("nan")),
        metrics.get("test_f1",   float("nan")),
        metrics.get("test_prec", float("nan")),
        metrics.get("test_rec",  float("nan")),
        metrics.get("test_auc",  float("nan")),
        metrics.get("train_acc", float("nan")),
        metrics.get("overfit",   float("nan")),
        metrics.get("tp", 0), metrics.get("fp", 0),
        metrics.get("fn", 0), metrics.get("tn", 0),
        metrics.get("sv_test_acc",  float("nan")),
        metrics.get("sv_test_f1",   float("nan")),
        metrics.get("sv_train_acc", float("nan")),
        metrics.get("ah_test_acc",  float("nan")),
        metrics.get("ah_test_f1",   float("nan")),
        metrics.get("as_test_acc",  float("nan")),
        metrics.get("as_test_f1",   float("nan")),
        runtime, status, error,
    ))
    con.commit()


# ================================================================
# SILENCE HELPER
# ================================================================

def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


# ================================================================
# TIMEOUT ENFORCEMENT (multiprocessing-based killable worker)
# ================================================================

def _run_miner_fit(miner, Xsel_train, y_clean, original_feature_names):
    """Top-level picklable wrapper that runs the exact same silent fit call."""
    return silent(miner.fit, Xsel_train, y_clean,
                  original_feature_names=original_feature_names)


def _timeout_worker(queue, target_fn, target_args, target_kwargs):
    """Worker executed in a separate process so it can be forcibly terminated."""
    try:
        result = target_fn(*target_args, **target_kwargs)
        queue.put((result, None))
    except Exception as e:
        queue.put((None, e))


def run_with_timeout(fn, args, kwargs, timeout_s):
    """
    Run fn(*args, **kwargs) in a background PROCESS (now truly killable).
    Returns (result, timed_out, exception).
      - If fn completes in time: (result, False, None)
      - If timeout:              (None,   True,  None)  ← process is TERMINATED
      - If exception:            (None,   False, exc)
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_timeout_worker,
        args=(queue, fn, args, kwargs)
    )
    process.start()

    process.join(timeout=timeout_s)

    if process.is_alive():
        # Enforced kill: the long-running miner is now actually terminated
        process.terminate()
        process.join()   # ensure cleanup
        return None, True, None

    # Process finished normally
    if not queue.empty():
        result, exc = queue.get()
        if exc is not None:
            return None, False, exc
        return result, False, None
    else:
        return None, False, RuntimeError("Worker process failed to report result")


# ================================================================
# CLASSIFIERS (inline -- no import from main to avoid side effects)
# ================================================================

def predict_best(row, sorted_rules, mwt):
    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return rule["label"], 1.0
    best_label, best_ov, best_wt = None, -1.0, -1.0
    for rule in sorted_rules:
        if rule["weight"] < mwt:
            continue
        attrs, values = rule["attrs"], rule["values"]
        ov = sum(row[attrs[i]] == values[i]
                 for i in range(len(attrs))) / len(attrs)
        if ov > best_ov or (ov == best_ov and rule["weight"] > best_wt):
            best_ov = ov; best_wt = rule["weight"]; best_label = rule["label"]
    if best_label is not None:
        return best_label, best_ov
    return sorted_rules[0]["label"], 0.0


def predict_soft(row, sorted_rules, mwt):
    scores = {}
    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            scores[rule["label"]] = scores.get(rule["label"], 0.0) + rule["weight"]
    if not scores:
        return predict_best(row, sorted_rules, mwt)
    best = max(scores, key=lambda l: scores[l])
    total = sum(scores.values())
    return best, scores[best] / total if total > 0 else 0.0


def predict_assume(row, sorted_rules, default):
    flip = 1 - default
    for rule in sorted_rules:
        if rule["label"] != flip:
            continue
        attrs, values = rule["attrs"], rule["values"]
        if all(row[attrs[i]] == values[i] for i in range(len(attrs))):
            return flip, rule["weight"]
    return default, 1.0


def run_clf(sorted_rules, X, mwt, mode):
    labels, confs = [], []
    for row in X:
        if   mode == "best":    lbl, c = predict_best(row, sorted_rules, mwt)
        elif mode == "soft":    lbl, c = predict_soft(row, sorted_rules, mwt)
        elif mode == "healthy": lbl, c = predict_assume(row, sorted_rules, 0)
        elif mode == "sick":    lbl, c = predict_assume(row, sorted_rules, 1)
        labels.append(lbl); confs.append(c)
    return np.array(labels), np.array(confs)


# ================================================================
# EVALUATION
# ================================================================

def evaluate_miner(miner, X_tr, y_tr, X_te, y_te, mwt):
    if not miner.rules:
        return {}

    sr = sorted(miner.rules, key=lambda r: r["weight"], reverse=True)

    bm_te,  bm_c_te  = run_clf(sr, X_te, mwt, "best")
    bm_tr,  _        = run_clf(sr, X_tr, mwt, "best")
    sv_te,  sv_c_te  = run_clf(sr, X_te, mwt, "soft")
    sv_tr,  _        = run_clf(sr, X_tr, mwt, "soft")
    ah_te,  _        = run_clf(sr, X_te, mwt, "healthy")
    as_te,  _        = run_clf(sr, X_te, mwt, "sick")

    bm_acc  = accuracy_score(y_te, bm_te)
    bm_bal  = balanced_accuracy_score(y_te, bm_te)
    bm_f1   = f1_score(y_te, bm_te, zero_division=0)
    bm_prec = precision_score(y_te, bm_te, zero_division=0)
    bm_rec  = recall_score(y_te, bm_te, zero_division=0)
    try:    bm_auc = roc_auc_score(y_te, bm_c_te)
    except: bm_auc = float("nan")
    bm_tr_acc = accuracy_score(y_tr, bm_tr)
    tn, fp, fn, tp = confusion_matrix(y_te, bm_te).ravel()

    return {
        "n_rules":   len(miner.rules),
        "test_acc":  bm_acc,   "test_bal":  bm_bal,
        "test_f1":   bm_f1,    "test_prec": bm_prec,
        "test_rec":  bm_rec,   "test_auc":  bm_auc,
        "train_acc": bm_tr_acc,"overfit":   bm_tr_acc - bm_acc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sv_test_acc":  accuracy_score(y_te, sv_te),
        "sv_test_f1":   f1_score(y_te, sv_te, zero_division=0),
        "sv_train_acc": accuracy_score(y_tr, sv_tr),
        "ah_test_acc":  accuracy_score(y_te, ah_te),
        "ah_test_f1":   f1_score(y_te, ah_te, zero_division=0),
        "as_test_acc":  accuracy_score(y_te, as_te),
        "as_test_f1":   f1_score(y_te, as_te, zero_division=0),
    }


# ================================================================
# LEADERBOARD
# ================================================================

def print_leaderboard(con, top_n=20):
    rows = con.execute("""
        SELECT
            selector, miner, purity,
            bf_top_k, bf_min_support, bf_max_group_size, bf_interval_frac,
            h_weight, mp_min_unique_cov, eager_min_support, threshold,
            n_cutpoints, n_selected_feats, n_rules,
            test_acc, test_f1, sv_test_acc, ah_test_acc, as_test_acc,
            train_acc, overfit_gap,
            tp, fp, fn, tn, runtime_s
        FROM   trials
        WHERE  status = 'ok'
        ORDER  BY test_acc DESC, test_f1 DESC
        LIMIT  ?
    """, (top_n,)).fetchall()

    total   = con.execute("SELECT COUNT(*) FROM trials WHERE status='ok'").fetchone()[0]
    failed  = con.execute("SELECT COUNT(*) FROM trials WHERE status='error'").fetchone()[0]
    timeout = con.execute("SELECT COUNT(*) FROM trials WHERE status='timeout'").fetchone()[0]
    W = 122
    print("\n" + "=" * W)
    print(f"  LEADERBOARD  (top {min(top_n,total)} of {total} ok, "
          f"{failed} errors, {timeout} timeouts)")
    print("=" * W)
    print(f"  {'Selector':>16} {'Miner':>12} {'Pur':>4}  "
          f"{'topk':>4} {'sup':>3} {'grp':>3} {'hw':>3}  "
          f"{'cuts':>4} {'sel':>3} {'rul':>3}  "
          f"{'BestAcc':>7} {'F1':>6} {'SoftAcc':>7} "
          f"{'AsH':>6} {'AsS':>6}  "
          f"{'TrAcc':>6} {'Gap':>6}  "
          f"{'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  "
          f"{'Sec':>4}")
    print("-" * W)
    for r in rows:
        (sel, miner, pur,
         topk, sup, grp, iv, hw, mp_cov, e_sup, thr,
         cuts, feats, nrules,
         t_acc, t_f1, sv_acc, ah_acc, as_acc,
         tr_acc, gap, tp, fp, fn, tn, rt) = r
        print(
            f"  {sel:>16} {miner:>12} {pur:>4.2f}  "
            f"{int(topk or 0):>4} {int(sup or 0):>3} {int(grp or 0):>3} "
            f"{float(iv or 0):.2f} {int(hw or 0):>3} thr={int(thr or 0)}  "
            f"{int(cuts or 0):>4} {int(feats or 0):>3} {int(nrules or 0):>3}  "
            f"{t_acc or 0:>7.4f} {t_f1 or 0:>6.4f} {sv_acc or 0:>7.4f} "
            f"{ah_acc or 0:>6.4f} {as_acc or 0:>6.4f}  "
            f"{tr_acc or 0:>6.4f} {gap or 0:>+6.4f}  "
            f"{int(tp or 0):>3} {int(fp or 0):>3} "
            f"{int(fn or 0):>3} {int(tn or 0):>3}  "
            f"{rt or 0:>4.1f}"
        )
    print("=" * W)


# ================================================================
# MAIN SEARCH LOOP
# ================================================================

def run_tuner():
    df = pd.read_csv(DATA_PATH)
    X  = df.iloc[:, :-1].values
    y  = df.iloc[:, -1].values.astype(int)
    feature_names = df.columns[:-1].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    con = init_db(DB_PATH)

    total_trials = len(BINARIZER_GRID) * len(SELECTOR_GRID) * len(MINER_GRID)
    done_count   = con.execute("SELECT COUNT(*) FROM trials").fetchone()[0]

    print(f"\nLAD Staged Hyperparameter Tuner")
    print(f"  Binarizer configs : {len(BINARIZER_GRID)}")
    print(f"  Selector  configs : {len(SELECTOR_GRID)}")
    print(f"  Miner     configs : {len(MINER_GRID)}")
    print(f"  Total trials      : {total_trials}")
    print(f"  Already done      : {done_count}")
    print(f"  Remaining         : {total_trials - done_count}")
    print(f"  DB                : {DB_PATH}\n")

    trial_num     = done_count
    t_total_start = time.time()

    # ── Stage 1: Binarizer ───────────────────────────────────────────────────
    for b_idx, b_cfg in enumerate(BINARIZER_GRID):

        # Skip entire binarizer if all its trials are logged
        b_all_done = all(
            is_done(con, make_hash(b_cfg, s_cfg, m_cfg))
            for s_cfg in SELECTOR_GRID
            for m_cfg in MINER_GRID
        )
        if b_all_done:
            continue

        print(f"\n{'='*65}")
        print(f"BINARIZER {b_idx+1}/{len(BINARIZER_GRID)}  "
              f"top_k={b_cfg['bf_top_k']}  "
              f"min_sup={b_cfg['bf_min_support']}  "
              f"max_grp={b_cfg['bf_max_group_size']}")
        print(f"{'='*65}")

        try:
            binarizer = BruteForceBinarizer(
                mode                = "greedy",
                top_k_per_feature   = b_cfg["bf_top_k"],
                min_support         = b_cfg["bf_min_support"],
                min_interval_fraction = b_cfg["bf_interval_frac"],
                categorical_cols    = CATEGORICAL_COLS,
                max_group_size      = b_cfg["bf_max_group_size"],
            )
            Xbin_train = silent(
                binarizer.fit_transform, X_train, y_train,
                feature_names=feature_names
            )
            Xbin_test = binarizer.transform(X_test)
            bin_names = [binarizer.cutpoint_readable(c)
                         for c in binarizer.cutpoints]
            n_cuts = len(binarizer.cutpoints)
            print(f"  Binarized: {Xbin_train.shape}  ({n_cuts} cutpoints)")

        except Exception as e:
            print(f"  [ERROR] Binarizer failed: {e}")
            for s_cfg in SELECTOR_GRID:
                for m_cfg in MINER_GRID:
                    h = make_hash(b_cfg, s_cfg, m_cfg)
                    if not is_done(con, h):
                        log_trial(con, h, b_cfg, s_cfg, m_cfg,
                                  {}, 0.0, "error",
                                  f"Binarizer failed: {e}")
            continue

        # ── Stage 2: Selector ────────────────────────────────────────────────
        for s_idx, s_cfg in enumerate(SELECTOR_GRID):

            # Skip if all miner trials for this (binarizer, selector) are done
            s_all_done = all(
                is_done(con, make_hash(b_cfg, s_cfg, m_cfg))
                for m_cfg in MINER_GRID
            )
            if s_all_done:
                continue

            sel_name = s_cfg["selector"]
            hw       = s_cfg["h_weight"]
            print(f"\n  SELECTOR {s_idx+1}/{len(SELECTOR_GRID)}: "
                  f"{sel_name}  h_weight={hw}  "
                  f"(max_expansions={ASTAR_MAX_EXPANSIONS})")

            try:
                if sel_name == "greedy":
                    selector = GreedyLADSelector()
                    selector.fit(Xbin_train, y_train, bin_names)
                elif sel_name == "astar":
                    selector = AStarFeatureSelector(
                        h_weight=hw,
                        max_expansions=ASTAR_MAX_EXPANSIONS,
                    )
                    selector.fit(Xbin_train, y_train, bin_names)
                elif sel_name == "mutualinfo":
                    selector = MutualInfoGreedySelector()
                    selector.fit(Xbin_train, y_train)
                elif sel_name == "mutualinfo_astar":
                    selector = MutualInfoAStarSelector(
                        h_weight=hw,
                        max_expansions=ASTAR_MAX_EXPANSIONS,
                    )
                    selector.fit(Xbin_train, y_train, bin_names)

                if not selector.best_subset:
                    raise ValueError("Empty feature subset")

                Xsel_train = selector.X_clean[:, selector.best_subset]
                y_clean    = selector.y_clean
                Xsel_test  = Xbin_test[:, selector.best_subset]
                n_sel      = len(selector.best_subset)
                print(f"    Selected {n_sel} features")

            except Exception as e:
                print(f"    [ERROR] Selector failed: {e}")
                for m_cfg in MINER_GRID:
                    h = make_hash(b_cfg, s_cfg, m_cfg)
                    if not is_done(con, h):
                        log_trial(con, h, b_cfg, s_cfg, m_cfg,
                                  {"n_cutpoints": n_cuts}, 0.0,
                                  "error", f"Selector failed: {e}")
                continue

            # ── Stage 3: Miner ───────────────────────────────────────────────
            for m_cfg in MINER_GRID:
                h = make_hash(b_cfg, s_cfg, m_cfg)
                if is_done(con, h):
                    continue

                trial_num  += 1
                miner_name  = m_cfg["miner"]
                purity      = m_cfg["purity"]
                mwt         = m_cfg["min_weight_thr"]

                print(f"    [{trial_num}/{total_trials}] "
                      f"{miner_name}  purity={purity}  ", end="", flush=True)

                t0 = time.time()
                try:
                    # Build miner object (fast, no fitting yet)
                    if miner_name == "maxpatterns":
                        miner = MaxPatterns(
                            binarizer           = binarizer,
                            selector            = selector,
                            purity              = purity,
                            verbose             = False,
                            threshold           = m_cfg["threshold"],
                            min_unique_coverage = m_cfg["mp_min_unique_cov"],
                        )
                    elif miner_name == "eager":
                        miner = Eager(
                            binarizer        = binarizer,
                            selector         = selector,
                            purity           = purity,
                            verbose          = False,
                            threshold        = m_cfg["threshold"],
                            min_rule_support = m_cfg["eager_min_support"],
                            n_rules_cap      = 100,
                        )

                    # === ENFORCED KILL LOGIC ===
                    # The fit now runs in a separate process that can be terminated
                    _, timed_out, fit_exc = run_with_timeout(
                        _run_miner_fit,
                        (miner, Xsel_train, y_clean, feature_names),
                        {},
                        TRIAL_TIMEOUT_SECONDS
                    )

                    if timed_out:
                        rt = time.time() - t0
                        log_trial(con, h, b_cfg, s_cfg, m_cfg,
                                  {"n_cutpoints": n_cuts, "n_selected": n_sel},
                                  rt, "timeout",
                                  f"Exceeded {TRIAL_TIMEOUT_SECONDS}s limit")
                        print(f"TIMEOUT ({rt/60:.1f} min) -- skipping")
                        continue

                    if fit_exc is not None:
                        raise fit_exc

                    if not miner.rules:
                        raise ValueError("python generated")

                    metrics = evaluate_miner(
                        miner, Xsel_train, y_clean,
                        Xsel_test, y_test, mwt
                    )
                    metrics["n_cutpoints"] = n_cuts
                    metrics["n_selected"]  = n_sel

                    rt = time.time() - t0
                    log_trial(con, h, b_cfg, s_cfg, m_cfg,
                              metrics, rt, "ok")
                    print(f"acc={metrics['test_acc']:.4f}  "
                          f"f1={metrics['test_f1']:.4f}  "
                          f"rules={metrics['n_rules']}  "
                          f"({rt:.1f}s)")

                except Exception as e:
                    rt = time.time() - t0
                    log_trial(con, h, b_cfg, s_cfg, m_cfg,
                              {"n_cutpoints": n_cuts, "n_selected": n_sel},
                              rt, "error", str(e))
                    print(f"ERROR: {e}")

            # ── Cleanup: free selector memory before next selector run ────
            try:
                del selector, Xsel_train, Xsel_test, y_clean
            except NameError:
                pass
            gc.collect()

        # ── Cleanup: free binarizer memory before next binarizer run ─────
        try:
            del binarizer, Xbin_train, Xbin_test, bin_names
        except NameError:
            pass
        gc.collect()

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total_start
    print(f"\nSearch complete in {elapsed/60:.1f} minutes.")
    print_leaderboard(con, top_n=20)

    # Export top 5 as JSON for copy-paste into main.py
    top = con.execute("""
        SELECT selector, miner, purity,
               bf_top_k, bf_min_support, bf_max_group_size, bf_interval_frac,
               h_weight, mp_min_unique_cov, eager_min_support,
               threshold, min_weight_thr, test_acc, test_f1,
               sv_test_acc, ah_test_acc, as_test_acc,
               n_rules, overfit_gap
        FROM   trials
        WHERE  status = 'ok'
        ORDER  BY test_acc DESC, test_f1 DESC
        LIMIT  5
    """).fetchall()

    fields = ["selector","miner","purity",
              "bf_top_k","bf_min_support","bf_max_group_size","bf_interval_frac",
              "h_weight","mp_min_unique_cov","eager_min_support",
              "threshold","min_weight_thr","test_acc","test_f1",
              "sv_test_acc","ah_test_acc","as_test_acc",
              "n_rules","overfit_gap"]

    print("\nTOP 5 CONFIGS  (copy best settings into main.py)")
    print("-" * 60)
    for row in top:
        print(json.dumps(dict(zip(fields, row)), indent=2))
        print()

    con.close()


if __name__ == "__main__":
    run_tuner()