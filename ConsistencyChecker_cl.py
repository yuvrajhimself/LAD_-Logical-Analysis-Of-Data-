import numpy as np
from collections import Counter


def check_consistency(X, y, subset):
    """
    Returns True if the given feature subset is consistent:
    no two rows that are identical across 'subset' columns have different labels.
    """
    if len(subset) == 0:
        return False
    Xsub = X[:, subset]
    pattern_labels = {}
    for pattern, label in zip([tuple(row) for row in Xsub], y):
        if pattern not in pattern_labels:
            pattern_labels[pattern] = label
        elif pattern_labels[pattern] != label:
            return False
    return True


def find_inconsistent_pairs(X, y, subset):
    """
    Returns a list of (pattern, labels_seen) for all conflicting patterns
    in the given subset. Used for logging.
    """
    if len(subset) == 0:
        return []
    Xsub = X[:, subset]
    pattern_labels = {}
    for pattern, label in zip([tuple(r) for r in Xsub], y):
        pattern_labels.setdefault(pattern, set()).add(int(label))

    return [
        (pattern, labels)
        for pattern, labels in pattern_labels.items()
        if len(labels) > 1
    ]


def remove_conflicting_rows(X, y, subset, verbose=True):
    """
    When full consistency is impossible (identical feature rows with different labels),
    logs the conflicts and removes the minority-label rows for each conflicting pattern.
    Majority label is kept; ties go to the first label encountered.

    Returns:
        X_clean, y_clean, removed_count
    """
    if len(subset) == 0:
        return X, y, 0

    Xsub = X[:, subset]
    rows_as_tuples = [tuple(row) for row in Xsub]

    # Map pattern -> list of (original_index, label)
    pattern_map = {}
    for i, (pattern, label) in enumerate(zip(rows_as_tuples, y)):
        pattern_map.setdefault(pattern, []).append((i, int(label)))

    keep_mask = np.ones(len(y), dtype=bool)
    removed_count = 0

    for pattern, entries in pattern_map.items():
        if len(set(lbl for _, lbl in entries)) == 1:
            continue  # consistent, keep all

        label_counts = Counter(lbl for _, lbl in entries)
        majority_label = label_counts.most_common(1)[0][0]
        conflicting_labels = sorted(set(lbl for _, lbl in entries))
        indices = [i for i, _ in entries]

        if verbose:
            print(
                f"[ConsistencyWarning] Pattern {pattern} maps to labels {conflicting_labels} "
                f"across {len(entries)} rows (indices {indices}). "
                f"Keeping majority label={majority_label}, removing minority rows."
            )

        for i, lbl in entries:
            if lbl != majority_label:
                keep_mask[i] = False
                removed_count += 1

    if removed_count > 0 and verbose:
        print(
            f"[ConsistencyWarning] Removed {removed_count} conflicting row(s) from training data. "
            f"{keep_mask.sum()} rows remain.\n"
        )

    return X[keep_mask], y[keep_mask], removed_count