# Model Selection — Comment Category Prediction

## Evaluation Metric

**Macro F1 Score** — chosen over accuracy because of significant class imbalance. Macro F1 averages the per-class F1 scores, giving equal weight to minority classes (1 and 3). A model that ignores class 3 entirely would still score well on accuracy but would fail on Macro F1.

All models were trained with `class_weight='balanced'` to counteract imbalance during optimization.

---

## Models Explored

### 1. Logistic Regression (Final ensemble member)

**Input:** Full sparse matrix — word TF-IDF + char TF-IDF + scaled base features (~70k+ features)

**Hyperparameter Tuning (GridSearchCV, 3-fold StratifiedKFold):**

Grid searched over:

```
C:        [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
tol:      [1e-4, 1e-3, 1e-2]
max_iter: [300, 500, 1000]
```

**Best Parameters found:**

```
C        = 3.0
max_iter = 300
tol      = 1e-3
```

**Final configuration:**

```python
LogisticRegression(
    C=3.0,
    class_weight='balanced',
    solver='liblinear',      # efficient for sparse high-dim data
    max_iter=500,
    multi_class='ovr',       # one-vs-rest for multiclass
    tol=1e-3,
    random_state=42
)
```

**Performance:**

- Strong on majority classes (0 and 2)
- Correct majority of the time on class 0 and 2
- **Weakness:** Low precision on class 3 (minority/toxic class) — not classifying it correctly with confidence

**Why it works well here:**
Logistic Regression thrives on large, sparse TF-IDF matrices. It efficiently learns linear separability in the high-dimensional token space — the dominant signal for text classification.

---

### 2. LinearSVC [x] (Tried, did not make the cut)

**Input:** Same full sparse matrix as Logistic Regression.

```python
LinearSVC(class_weight='balanced', C=0.1, max_iter=1000, tol=1e-3)
```

Since LinearSVC doesn't natively output probabilities (needed for soft-voting ensemble), it was wrapped with `CalibratedClassifierCV` using isotonic regression:

```python
CalibratedClassifierCV(lsvc, cv=3, method='isotonic')
```

**Result:** Could not break past **0.72 Macro F1** on validation. Abandoned in favour of Logistic Regression which provided better probability calibration and marginally better scores.

---

### 3. SGDClassifier [x] (Tried, abandoned)

```python
SGDClassifier(
    loss='log_loss',        # equivalent to online logistic regression
    penalty='l2',
    alpha=1e-4,
    class_weight='balanced',
    early_stopping=True
)
```

Used `log_loss` to get probabilistic outputs. Performance was not competitive enough with the tuned Logistic Regression. Code remains commented out in the notebook. No specific F1 number recorded in observations.

---

### 4. HistGradientBoostingClassifier (Used in ensemble with zero weight)

**Input:** Dense SVD-reduced matrix (300 word SVD + 150 char SVD + base features)

```python
HistGradientBoostingClassifier(
    learning_rate=0.05,
    class_weight='balanced',
    max_iter=1000,
    max_leaf_nodes=31,
    random_state=42
)
```

**Why SVD instead of sparse?**
HGB requires a dense input. Raw TF-IDF is ~70k features — too wide and memory-intensive for a dense representation. TruncatedSVD compressed this to 450 latent dimensions while preserving the majority of variance.

**Performance:**
HGB was trained and predicted probabilities were collected (`probs_val_hgb`). However, in the final ensemble, its weight was set to **0.0** — meaning it was ultimately not included in the voting. The tree-based non-linear learning on SVD features did not add enough above LightGBM's contribution to justify inclusion. Code remains functional if the weight is increased.

---

### 5. LightGBM (LGBMClassifier) (Final ensemble member)

**Input:** Full sparse matrix — word TF-IDF + char TF-IDF + scaled base features (same as Logistic Regression).

LightGBM natively supports **sparse input**, making it ideal for TF-IDF matrices without requiring SVD dimensionality reduction. It can also learn the interaction between TF-IDF token signals and the manually engineered base features.

**Configuration explored (commented):**

```python
# Heavier config (explored)
LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight='balanced',
)
```

**Final configuration used:**

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=50,
    min_child_samples=20,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

Training used `early_stopping(100)` on the validation set to prevent overfitting — training stops if validation metric doesn't improve for 100 rounds.

**Why LightGBM here:**
It complements Logistic Regression by capturing **non-linear interactions** in both the text space and structured features. Where Logistic Regression draws straight hyperplanes in feature space, LightGBM builds decision trees that capture patterns like "IF if_2==10 AND has_threat_word==1 THEN label=3" which can't be expressed linearly.

---

## Ensemble Strategy: Soft Voting (Probability Blending)

The final prediction is a **weighted average of predicted class probabilities** from multiple models — also called soft voting or probability blending.

### Rationale

Each model type captures different aspects of the data:

- **Logistic Regression** → strong linear signal from 50k token space (TF-IDF bigrams + char n-grams)
- **LightGBM** → non-linear interactions in token space + structured features
- **HGB** → non-linear learning on compressed latent SVD representations

Blending allows each model's confidence to vote proportionally — a model that's very confident in class 3 will outweigh a model that's hedging.

### Final Weights

```python
w_log = 0.6
w_lgb = 0.4
w_hgb = 0.0    # not included in final submission

final_probs = (0.6 * probs_logreg) + (0.4 * probs_lgbm)
y_pred = np.argmax(final_probs, axis=1)
```

The **60/40 split** reflects Logistic Regression's stronger performance on majority classes balanced against LightGBM's better handling of non-linear patterns in minority classes. HGB was zeroed out because its contribution over LightGBM was marginal.

---

## Model Performance Summary

| Model                        | Val Macro F1 | Notes                            |
| ---------------------------- | ------------ | -------------------------------- |
| LinearSVC                    | < 0.72       | Capped, calibration issues       |
| SGDClassifier                | Not recorded | Abandoned early                  |
| HGB                          | —            | Included but weight = 0 in final |
| Logistic Regression          | ~0.7x        | Best on classes 0 & 2            |
| LightGBM                     | ~0.7x        | Better on minority classes       |
| **Ensemble (LogReg + LGBM)** | **Best**     | Submission model                 |

---

## Key Observations on Classification Behaviour

**Class 0 and 2 (majority/political):** Both Logistic Regression and LightGBM handled these well. High recall and precision.

**Class 1 (racism — minority):** Moderate performance. Identity word features and TF-IDF bigrams help, but limited training examples make generalisation harder.

**Class 3 (toxic threats — minority):** Hardest class. Low precision in Logistic Regression — frequently confused with class 1 or 2. Threat word lexical features (`has_threat_word`, `threat_word_density`) were intended to help here, but the class is rare enough that even with `class_weight='balanced'`, confidence is lower.

---

## Milestone Experiments (Structured Questions)

Separate from the main pipeline, the notebook also explored a series of structured milestone experiments using different preprocessing pipelines and models for evaluation purposes. These are documented here:

### Milestone 3 — MultinomialNB baseline

- Basic pipeline: TF-IDF + OHE + numeric imputation → MultinomialNB
- `np.abs()` applied to numeric features before MNB (MNB requires non-negative input)
- Used as a sanity-check baseline — not intended as a competitive model

### Milestone 4 — RandomForest, AdaBoost, MLP

- Input: TruncatedSVD (300 components) of the full feature matrix
- RandomForest tuned via RandomizedSearchCV — explored `n_estimators` and `max_depth`
- AdaBoost: `estimator_errors_` variance computed as a stability diagnostic
- MLP: Architecture `(128, 64, 32)` with ReLU; compared regularization strengths (`alpha=0.0001` vs `alpha=1.0`)
- These were milestone/graded exercises, not part of the submission pipeline

### Milestone 5 — GridSearchCV on Logistic Regression

- Used `C = [0.1, 1, 10]` with `liblinear` solver and 3-fold CV
- Confirmed the direction taken in the main pipeline: higher C (less regularization) performs better on this dataset

---

## What Worked vs What Didn't

| Decision                                    | Outcome                                            |
| ------------------------------------------- | -------------------------------------------------- |
| Logistic Regression on sparse TF-IDF        | ✅ Strong linear baseline for text                 |
| Character n-gram TF-IDF in ensemble         | ✅ Adds subword/morphological signals              |
| LightGBM on full sparse matrix              | ✅ Captures non-linear feature interactions        |
| Soft voting (probability blending)          | ✅ Smooths out individual model weaknesses         |
| `class_weight='balanced'` across all models | ✅ Essential for minority class recall             |
| LinearSVC                                   | ❌ Plateaued below 0.72 Macro F1                   |
| SGDClassifier                               | ❌ Not competitive vs tuned LogReg                 |
| HGB in ensemble                             | ❌ Marginal gain; weight set to 0.0                |
| GridSearchCV hyperparameter tuning (LogReg) | ✅ Found C=3 significantly better than default C=1 |
| Early stopping in LightGBM                  | ✅ Prevented overfitting on a 300-tree config      |
