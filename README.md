# LAD Heart Disease Classification Pipeline

This project performs pixel-wise — actually patient-wise — heart disease classification on the Cleveland Heart Disease dataset using **Logical Analysis of Data (LAD)**. It includes binarization of continuous and categorical features, binary feature selection, rule mining, and multi-classifier evaluation.

The goal is to learn interpretable IF-THEN rules that classify patients as healthy or diseased, producing a transparent, auditable model suitable for clinical reasoning.

---

## Features

> False-color binarization via BruteForceBinarizer (numerical + categorical cutpoints)  
> Decision-tree-based binarizer as an alternative (DecisionTreeCutpointBinarizerV2)  
> Four feature selector strategies: Greedy, A\*, MutualInfo-Greedy, MutualInfo-A\*  
> LAD consistency checking — ensures no two identical binary patterns map to different labels  
> Four rule miners: MaxPatterns, Eager (sequential covering), Genetic Algorithm, Lazy (instance-based)  
> Four classifiers compared side-by-side: Best Match, Soft Vote, Assume Healthy, Assume Sick  
> Full metrics suite: Accuracy, Balanced Accuracy, F1, Precision, Recall, AUC, Confusion Matrix  
> Optional purity and A\* h\_weight auto-tuning via stratified cross-validation  
> Model export via pickle; GeoTIFF-equivalent output as classified rule maps  

---

## Tech Stack

> Python  
> scikit-learn  
> numpy / pandas  
> joblib / pickle  
> matplotlib  

---

## Project Structure

```
Heart_disease_cleveland_new.csv          input tabular dataset
main_cl.py                               main pipeline script
BruteForceBinarizer_cl.py                numerical + categorical binarizer
DecisionTreeCutpointBinarizerV2.py       decision-tree cutpoint binarizer
AStarFeatureSelector_cl.py               A* best-first binary feature selector
GreedyLADSelector_cl.py                  forward stepwise greedy selector
MutualInfoAStarSelector_cl.py            MI pre-ranked A* selector
MutualInfoGreedySelector_cl.py           MI pre-ranked greedy selector
LADScorer_cl.py                          shared coverage-weighted purity scorer
ConsistencyChecker_cl.py                 conflict detection and row removal
MaxPatterns_cl.py                        MaxPatterns eager rule miner
Eager_cl.py                              sequential covering rule learner
GeneticRuleMiner_cl.py                   genetic algorithm rule miner
LazyPatterns_cl.py                       lazy instance-based rule learner
tuner_cl.py                              purity and h_weight tuning utilities
new_data_main_cl.py                      pipeline entry for new/unseen data
rf_model.joblib                          saved model (generated at runtime)
requirements.txt                         dependencies
README.md
```

---

## Land Cover — Heart Disease Classes

| Class ID | Name    | Meaning                          |
|----------|---------|----------------------------------|
| 0        | Healthy | No heart disease detected        |
| 1        | Disease | Presence of heart disease        |

---

## Feature Set (Cleveland Dataset — 13 Features)

| # | Feature   | Type        | Description                              |
|---|-----------|-------------|------------------------------------------|
| 0 | age       | Numerical   | Age in years                             |
| 1 | cp        | Categorical | Chest pain type (0–3)                    |
| 2 | fbs       | Categorical | Fasting blood sugar > 120 mg/dl          |
| 3 | trestbps  | Numerical   | Resting blood pressure                   |
| 4 | chol      | Numerical   | Serum cholesterol in mg/dl               |
| 5 | restecg   | Categorical | Resting ECG results (0–2)                |
| 6 | exang     | Categorical | Exercise-induced angina                  |
| 7 | thalach   | Numerical   | Maximum heart rate achieved              |
| 8 | slope     | Categorical | Slope of peak exercise ST segment        |
| 9 | oldpeak   | Numerical   | ST depression induced by exercise        |
| 10| ca        | Categorical | Number of major vessels colored (0–3)    |
| 11| thal      | Categorical | Thalassemia type (3=normal, 6=fixed, 7=reversable) |
| 12| thalach   | Numerical   | Maximum heart rate achieved              |

---

## Pipeline

1. **Load & Split** — CSV loaded via pandas; stratified 80/20 train/test split preserving class balance
2. **Binarize** — Continuous features converted to binary columns via cutpoints (BruteForce or DecisionTree); categorical columns handled via equality/grouping cuts scored by Gini gain
3. **Feature Selection** — From potentially 100–300 binary columns, a minimal consistent subset is found using LAD score (coverage-weighted purity); four strategies available
4. **Consistency Check** — Conflicting rows (same binary pattern, different labels) are detected and minority-label rows removed before mining
5. **Rule Mining** — IF-THEN rules mined from the selected binary matrix with configurable purity and support thresholds; four mining strategies available
6. **Classify & Evaluate** — All four classifiers run on train and test sets; side-by-side metrics table printed
7. **Optional Tuning** — A\* h\_weight and purity threshold auto-selected via stratified CV if enabled

---

## How to Run

```bash
git clone <repo-link>
cd <repo-name>
pip install -r requirements.txt
python main_cl.py
```

---

## Configuration (top of `main_cl.py`)

| Parameter               | Default           | Description                                      |
|-------------------------|-------------------|--------------------------------------------------|
| `BINARIZER_TYPE`        | `"bruteforce"`    | `"bruteforce"` or `"decisiontree"`              |
| `SELECTOR`              | `"mutualinfo_astar"` | Selector strategy                             |
| `PATTERN_MINER`         | `"maxpatterns"`   | Mining strategy                                  |
| `PURITY`                | `0.95`            | Minimum rule purity threshold (0.6–1.0)          |
| `THRESHOLD`             | `3`               | Minimum rows a rule must cover                   |
| `TUNE_PURITY`           | `False`           | Auto-tune purity via 5-fold CV                   |
| `TUNE_ASTAR`            | `False`           | Auto-tune A\* h\_weight                          |
| `ASTAR_H_WEIGHT`        | `5`               | Purity vs. subset-size trade-off in A\*          |
| `MAX_PATTERNS_MIN_UNIQUE_COV` | `2`         | Redundancy pruning threshold for MaxPatterns     |
| `GA_GENERATIONS`        | `300`             | Generations for genetic miner                    |
| `GA_POP_SIZE`           | `200`             | Population size for genetic miner                |
| `RUN_CV`                | `False`           | Run full pipeline cross-validation after main eval |

---

## Requirements

```
numpy
pandas
scikit-learn
joblib
matplotlib
```

---

## Results

The LAD pipeline produces human-readable IF-THEN rules for binary heart disease classification. Four classifiers are evaluated on the same rule set:

- **Best Match** — exact rule match, then strongest partial overlap, then fallback
- **Soft Vote** — all firing rules vote by weight; class with highest total weight wins
- **Assume Healthy** — defaults to 0 (no disease); flips only if a disease rule fires exactly
- **Assume Sick** — defaults to 1 (disease); flips only if a healthy rule fires exactly

Example rule output:
```
Rule  1: IF cp IN {2,3} AND thalach <= 153.5 THEN class = 1
         | Purity=0.950 | Support=19 | Weight=0.613

Rule  2: IF NOT (ca IN {1}) AND age <= 54.5 THEN class = 0
         | Purity=1.000 | Support=14 | Weight=0.560
```

---

## Known Limitations

- Affine/geotransform concepts do not apply — no spatial output, purely tabular
- `n_estimators` equivalent is `PURITY` and `THRESHOLD` — low purity produces noisy rules; tune carefully
- Consistency removal may drop conflicting training rows — monitor `[ConsistencyWarning]` output
- Genetic miner with 300 generations on large binary matrices can be slow; reduce `GA_GENERATIONS` if needed
- `LazyPatterns` has no upfront rules — cannot be persisted the same way as eager miners

---

## Future Improvements

> Replace hardcoded categorical column indices with auto-detection from dtypes  
> Add SHAP-style feature importance plot over selected binary features  
> Extend to multi-class (original 5-level target instead of binarized)  
> Add confusion matrix heatmap visualization  
> Compare LAD rules against LIME/SHAP explanations on the same dataset  
> Deploy via Streamlit — enter patient vitals, get rule-based prediction with justification  
> Support batch prediction on new patient CSV files via `new_data_main_cl.py`  
> Benchmark against SVM, Logistic Regression, and XGBoost on the same split  

---

## Disclaimer

This project is for research and educational purposes only. The rules and predictions produced by this pipeline are not validated clinical tools and must not be used for actual medical diagnosis without expert review and proper validation.
# LAD_-Logical-Analysis-Of-Data-
ML model
