# Overview — Comment Category Prediction Challenge

## Problem Statement

Classify online comments into one of **4 categories** (labels 0–3) using a mix of the raw comment text and structured metadata (votes, engagement signals, demographic information).

Based on EDA and word cloud analysis, the labels map roughly to:
| Label | Inferred Category |
|-------|------------------|
| 0 | Neutral / Political |
| 1 | Racist / Identity-based |
| 2 | Political (overlapping with 0) |
| 3 | Toxic / Threatening |

This is a real-world noisy text classification problem with **class imbalance**, **high dimensionality**, and **heterogeneous input types** (raw text + structured features + sparse demographics).

---

## The Core Challenge

Three factors make this non-trivial:

**1. Class Imbalance**
Labels 1 and 3 are minority classes. A naive model that always predicts class 0 would achieve high accuracy but be useless. This immediately rules out accuracy as a metric — **Macro F1** was used throughout, and `class_weight='balanced'` was applied to every model.

**2. Heterogeneous Features**
The dataset isn't just text. It contains:
- Raw comment text → needs NLP (TF-IDF vectorization)
- Numeric engagement signals (`upvote`, `downvote`, `if_1`, `if_2`) → need scaling
- Categorical demographics (`race`, `religion`, `gender`) → 73%+ missing, need careful handling
- Boolean (`disability`) → needs integer mapping

A good solution needs to combine all of these coherently.

**3. Minority Class Difficulty**
Class 3 (toxic/threats) is both rare and hard to identify. Even with balancing, models consistently underperformed on class 3 precision. Lexical features (threat word lists) were specifically engineered to help, but the challenge remains.

---

## Thinking Behind the Approach

### Step 1 — Understand the data before touching it

Significant EDA was done before writing a single preprocessing line:
- Correlation heatmaps revealed `if_2` as a strong predictor
- Word clouds identified the thematic character of each class
- Class distribution plots confirmed the need for F1 over accuracy
- Missing value analysis led to the `"unknown"` category decision for demographics

This meant feature engineering was **targeted**, not random.

### Step 2 — Extract signal from multiple information sources simultaneously

The key insight: the comment text alone is not enough. A single comment might look neutral but have an `if_2 == 10` value that strongly signals a category, or a demographic profile that shifts probability. The pipeline was designed to combine:

- **Text signal** (what is said) via TF-IDF word and character n-grams
- **Structural signal** (how it was said) via text statistics — length, casing, punctuation
- **Engagement signal** (how the community responded) via votes and interaction features
- **Context signal** (who said it) via demographic OHE columns

### Step 3 — Use complementary models

No single model is best at everything here:
- **Logistic Regression** is fast and excellent at high-dimensional sparse data (TF-IDF). It learns which words and bigrams are predictive of each class.
- **LightGBM** captures non-linear interactions — IF this token AND this engagement pattern THEN this class — which a linear model structurally cannot do.
- **Soft-voting ensemble** blends their probability outputs weighted by confidence: Logistic Regression at 60%, LightGBM at 40%.

This is a classic **bias-variance trade-off** play: LR has higher bias (linear), LGBM has higher variance (overfits more). Blending reduces both.

---

## Pipeline Architecture

```
Raw Data (train.csv / test.csv)
          │
          ▼
    Data Audit & EDA
    - Missing value analysis
    - Class distribution
    - Correlation heatmaps
    - Word clouds per label
          │
          ▼
    Feature Engineering (preprocess())
    ├── Text stats (word_count, upper_ratio, ...)
    ├── Date decomposition (hour, day_of_week)
    ├── Vote interactions (total_votes, net_sentiment, ...)
    ├── if_* interactions (if_product, if2_is_10, ...)
    ├── Lexical features (identity/threat/political word counts)
    ├── Text cleaning (clean_text)
    └── Categorical encoding (OHE via ColumnTransformer)
          │
          ▼
    ┌─────────────────────────────────────────┐
    │         Two feature matrices            │
    │                                         │
    │  Sparse (for LR + LGBM):               │
    │  word TF-IDF + char TF-IDF + base      │
    │  (~70,000 features)                     │
    │                                         │
    │  Dense SVD (for HGB):                  │
    │  300 SVD + 150 char SVD + base         │
    │  (~500 features)                        │
    └─────────────────────────────────────────┘
          │
          ▼
    Model Training (80/20 stratified split)
    ├── Logistic Regression (tuned: C=3, liblinear)
    ├── LightGBM (300 trees, early stopping)
    └── HistGradientBoosting (trained, weight=0 in final)
          │
          ▼
    Soft-Vote Ensemble
    final = 0.6 × P(LogReg) + 0.4 × P(LGBM)
          │
          ▼
    Submission
```

---

## Key Decisions & Rationale

| Decision | Why |
|----------|-----|
| Macro F1 as metric | Class imbalance makes accuracy misleading |
| `class_weight='balanced'` everywhere | Prevents minority class suppression |
| `"unknown"` for demographic nulls | 73%+ MAR missingness — imputing would be fabrication |
| OHE over Label Encoding for demographics | No ordinal relationship between race/religion/gender categories |
| Character n-gram TF-IDF (2–5) | Captures obfuscated spellings, morphological variants |
| Word bigram TF-IDF (1–2) | Phrases like "death threat" more informative than individual tokens |
| `sublinear_tf=True` in word TF-IDF | Dampens effect of extremely high-frequency tokens |
| Threat/identity/political word dictionaries | Word cloud EDA directly informed these |
| `if2_is_10` binary flag | EDA revealed `if_2 == 10` heavily correlated with target |
| SVD for HGB | HGB doesn't accept sparse; 70k→450 dense features |
| Ensemble over single model | Captures both linear text patterns and non-linear interactions |
| Early stopping in LGBM (100 rounds) | Prevents memorizing training distribution |
| GridSearchCV for LogReg tuning | Found C=3 substantially better than default C=1 |

---

## What Was Tried and Abandoned

| Approach | Reason Abandoned |
|----------|-----------------|
| LinearSVC | Capped at ~0.72 Macro F1; no clear path to improvement |
| SGDClassifier | Not competitive with tuned Logistic Regression |
| Label Encoding for categorical cols | No ordinal relationship — meaningless encoding |
| `word_zscore` feature | Pure duplicate of `word_count` |
| `char_count` feature | 0.99 correlation with `word_count` — redundant |
| `hour` and `day_of_week` | Near-zero correlation with target |
| `vote_ratio` | Redundant with other vote features |
| HGB in the final ensemble | Marginal gain; weight set to 0.0 |

---

## Milestone Exercises

The notebook also contains 5 structured milestone question sets (likely for a graded competition track). These include:

- **Milestones 1–2:** Basic data exploration — shape, dtypes, missing values, text stats
- **Milestone 3:** Standard OHE + TF-IDF → MultinomialNB baseline
- **Milestone 4:** SVD → RandomForest / AdaBoost / MLP experiments; hyperparameter tuning with RandomizedSearchCV
- **Milestone 5:** Full preprocessing pipeline as specified → MNB baseline, LogReg with GridSearchCV tuning

These are separate from the submission pipeline and served as graded checkpoints and baseline sanity checks. The patterns learned here (what preprocessing steps to use, which models struggle on imbalanced text data) directly informed the final pipeline design.

---

## Files in This Repository

| File | Description |
|------|-------------|
| `preprocessing.md` | Detailed walkthrough of all feature engineering decisions, what was kept, what was dropped, and why |
| `model_selection.md` | All models tried, configurations, performance notes, ensemble design, and milestone experiments |
| `overview.md` | This file — high-level problem framing, thinking process, and full pipeline architecture |
| `notebook.ipynb` | Full working notebook with all code, observations, and commented experiments |

---

## Results

The final submission uses a **60/40 soft-vote ensemble of Logistic Regression and LightGBM** trained on a combined sparse matrix of word TF-IDF, character TF-IDF, and hand-engineered structured features. The ensemble was evaluated on a 20% stratified validation split using Macro F1 as the primary metric.
