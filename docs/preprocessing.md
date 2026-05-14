# Preprocessing — Comment Category Prediction

## Problem Context

This is a **multi-class text classification** task with 4 target labels (0, 1, 2, 3) representing comment categories — ranging from political/neutral to toxic/threatening. The dataset comes from the Kaggle competition `comment-category-prediction-challenge` and contains both text (`comment`) and structured metadata (votes, engagement signals, demographic info).

---

## 1. Initial Data Audit

### Shape & Structure
- `train.csv` and `test.csv` loaded from Kaggle input.
- All column dtypes were mostly correct out of the box — only `created_date` needed a datetime conversion.

### Missing Values
```
race       → 73%+ null
religion   → 73%+ null
gender     → 73%+ null
comment    → 1 null row (dropped)
```

**Observation:** The missingness in `race`, `religion`, and `gender` was classified as **Missing At Random (MAR)** — users intentionally withheld identity for privacy. This ruled out imputation with statistical values (mean/mode). Instead, a dedicated `"unknown"` category was used, preserving the signal that the user chose not to disclose.

### Descriptive Statistics
- Right-skewed distributions across most numeric columns — mean > median (classic positive skew).
- Outliers present in `emoticon_1`, `emoticon_2`, `emoticon_3`, `upvote`, `downvote`, `if_1`, `if_2`.
- Very large maximum values in `upvote`, `downvote`, `if_1`, `if_2`.
- **Implication:** Scaling was necessary before feeding into linear models (Logistic Regression). Tree-based models (LightGBM, HGB) are scale-invariant.

---

## 2. Exploratory Data Analysis

### Class Imbalance
The label distribution was highly imbalanced:
- **Labels 1 and 3 were minority classes.**
- **Label 0** dominated the dataset.
- **Consequence:** Accuracy was dropped as the primary metric in favour of **Macro F1**, which equally weights performance across all classes. `class_weight='balanced'` was used in all models.

### Text Length Distribution (per label)
- Word count distributions were **right-skewed** for all labels (mean > median).
- Outliers in long comments pull the mean rightward.
- No dramatic difference in comment length between labels — word count alone is not a separating signal.

### Correlation Heatmap (Pre-feature Engineering)
Key findings:
- `if_2` is **highly correlated with the target** — strongest numeric predictor.
- `downvote` and `emoticon_3` are inter-correlated.
- `downvote` and `upvote` are inter-correlated.

### Word Cloud Insights (per label)
Visual inspection of word clouds revealed clear thematic separation:
| Label | Theme |
|-------|-------|
| 0 | Political content |
| 1 | Racism / identity-based content |
| 2 | Political (overlapping with 0) |
| 3 | Toxic threats — violence, harm |

This guided the design of **lexical feature dictionaries** in the preprocessing step.

---

## 3. Feature Engineering

All features were created inside a single `preprocess()` function applied identically to train and test sets.

### 3.1 Text Statistics
| Feature | Description |
|---------|-------------|
| `word_count` | Number of whitespace-split tokens |
| `char_count` | Total character count |
| `char_count_log` | `log1p(char_count)` — corrects right skew |
| `upper_ratio` | Fraction of uppercase characters — proxy for shouting/aggression |
| `exclamation_count` | Count of `!` |
| `question_count` | Count of `?` |
| `ellipse_count` | Count of `...` — hesitation or trailing off |
| `word_zscore` | Z-score of word count |

**Post-heatmap findings (redundancy check):**
- `word_zscore` is a **pure duplicate** of `word_count` (same signal, different scale) → **dropped**.
- `char_count` correlates at **0.99** with `word_count` → **dropped** in final feature list (kept `char_count_log`).
- `ellipse_count`, `question_count`, `hour`, `day_of_week` showed **near-zero correlation** with target → excluded from base features.

### 3.2 Datetime Features
- `created_date` → converted to datetime, then decomposed into:
  - `hour` — time of day
  - `day_of_week` — day-level signal

Both showed near-zero correlation with the target and were not included in the final feature set.

### 3.3 Engagement / Vote Features
| Feature | Formula |
|---------|---------|
| `total_votes` | `upvote + downvote` |
| `vote_ratio` | `upvote / (downvote + 1)` |
| `net_sentiment` | `upvote - downvote` |

`vote_ratio` was dropped after the second correlation heatmap.

### 3.4 `if_*` Interaction Features
`if_1` and `if_2` were among the strongest numeric signals (especially `if_2`). Several interaction terms were engineered:

| Feature | Description |
|---------|-------------|
| `if_product` | `if_1 * if_2` |
| `if_ratio` | `if_1 / (if_2 + 1)` |
| `if_diff` | `if_1 - if_2` |
| `if_max` | `max(if_1, if_2)` |
| `if_total` | `if_1 + if_2` |
| `if2_is_10` | Binary — value 10 was the most frequent in `if_2` and highly correlated with target |
| `if2_is_4` | Binary — second most frequent value in `if_2` |
| `if1_nz_if2_10` | `if_1 != 0` AND `if_2 == 10` — joint condition, highly correlated with target |
| `if2_4_if1_0` | `if_2 == 4` AND `if_1 == 0` |
| `if_both` | Both `if_1 > 0` and `if_2 > 0` |

**Observation:** `if2_is_10` and `if1_nz_if2_10` both showed high correlation with the target in the post-engineering heatmap.

### 3.5 Lexical Keyword Features
Three manually curated word dictionaries were built based on word cloud insights:

**Identity Words:**
`muslim, black, white, christian, jewish, hindu, gay, transgender, women, racist, latino, religion, catholic`

**Threat Words:**
`kill, shot, shoot, dead, die, gun, death, murder, hurt, attack, bomb, threat, killing, killed, weapon, illegal, fire, lock, war, penalty`

**Political Words:**
`government, trump, president, republican, tax, state, america, american`

For each dictionary:
- `*_word_count` — how many matching words in comment
- `has_*_word` — binary flag
- `threat_word_density` — `threat_word_count / (word_count + 1)`

### 3.6 Text Cleaning
A `clean_text()` function was applied to produce a normalized `clean_text` column used as input to TF-IDF vectorizers:

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)                        # remove digits
    text = re.sub(r'\s+', ' ', text).strip()               # normalize whitespace
    return text
```

Note: Punctuation was **not removed** at this stage — it feeds into character-level n-gram TF-IDF which benefits from punctuation marks.

### 3.7 Categorical Encoding

**`disability`:** Boolean → mapped to `{True: 1, False: 0}`.

**`race`, `religion`, `gender`:**
- Nulls filled with `"unknown"` string (not dropped, not statistically imputed).
- **Label Encoding was tried first** (commented out in notebook) but abandoned in favour of **One-Hot Encoding** using `ColumnTransformer` with `drop='first'` to avoid dummy variable trap.
- `fit` was done only on train; `transform` applied to both train and test.

Final OHE columns produced:
- `race_*`: black, latino, none, other, unknown, white
- `gender_*`: male, none, other, transgender, unknown
- `religion_*`: buddhist, christian, hindu, jewish, muslim, none, other, unknown

---

## 4. Dataset Construction for Models

Two versions of the feature matrix were built — one for sparse linear models, one for dense tree-based models.

### 4.1 TF-IDF Vectorization

**Word-level TF-IDF:**
```
ngram_range = (1, 2)      # unigrams and bigrams
min_df = 3                 # ignore very rare terms
max_df = 0.95              # ignore near-universal terms
sublinear_tf = True        # log(1 + tf) — dampens high-frequency terms
stop_words = 'english'
max_features = 50,000
```

**Character-level TF-IDF:**
```
analyzer = 'char'
ngram_range = (2, 5)       # subword patterns — catches morphological variants
min_df = 5
max_features = 20,000
```

Character n-grams capture patterns like partial words, typos, and deliberate obfuscation (e.g., "k1ll" vs "kill").

### 4.2 Base Numeric Feature Matrix
Final selected features (after correlation analysis and redundancy pruning):

```python
base_features = [
    # Votes
    'upvote', 'downvote', 'total_votes', 'net_sentiment',

    # if columns + engineered interactions
    'if_1', 'if_2', 'if_max', 'if_total', 'if_ratio', 'if_diff',
    'if_product', 'if_both', 'if2_is_10', 'if2_is_4',
    'if1_nz_if2_10', 'if2_4_if1_0',

    # text statistics
    'word_count', 'char_count_log', 'exclamation_count', 'upper_ratio',

    # lexical features
    'identity_word_count', 'has_identity_word',
    'threat_word_count', 'threat_word_density', 'has_threat_word',
    'political_word_count', 'has_political_word',

    # demographics (OHE)
    'disability',
    'race_black', 'race_latino', 'race_none', 'race_other', 'race_unknown', 'race_white',
    'gender_male', 'gender_none', 'gender_other', 'gender_transgender', 'gender_unknown',
    'religion_buddhist', 'religion_christian', 'religion_hindu', 'religion_jewish',
    'religion_muslim', 'religion_none', 'religion_other', 'religion_unknown',
]
```

Scaled with `StandardScaler` and converted to a sparse matrix for compatibility with `scipy.sparse.hstack`.

### 4.3 Full Sparse Matrix (for linear models + LightGBM)
```
X_train_full = hstack([X_word_train, X_char_train, X_base_train])
Shape → (n_samples, ~70,000+ features)
```

### 4.4 Dense SVD Matrix (for HistGradientBoosting)
Tree-based models like HGB cannot natively handle sparse matrices. TruncatedSVD was used to project TF-IDF into a dense lower-dimensional space:

```
SVD word features:  300 components  (from 50,000)
SVD char features:  150 components  (from 20,000)
Final dense matrix: 300 + 150 + base_features
```

```python
X_train_full_hgb = np.hstack([X_train_text_svd, X_train_char_svd, X_base_train.toarray()])
```

---

## 5. Train-Validation Split

```python
train_idx, val_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=42,
    stratify=y      # preserves class distribution in both splits
)
```

`stratify=y` is critical here — without it, the minority classes (1 and 3) could be underrepresented in validation, giving misleadingly optimistic F1 scores.

---

## 6. What Worked vs What Didn't

| Decision | Outcome |
|----------|---------|
| `"unknown"` for MAR demographic nulls | ✅ Correct — preserves privacy signal |
| Dropping `word_zscore` (duplicate) | ✅ Cleaner feature space |
| Dropping `char_count` (0.99 corr with word_count) | ✅ Reduced redundancy |
| Character n-gram TF-IDF | ✅ Captures obfuscated/misspelled threat words |
| `if_2 == 10` binary flag | ✅ Strong signal — visible in both EDA and correlation heatmap |
| Label Encoding for categorical cols | ❌ Switched to OHE — nominal categories have no ordinal relationship |
| Including `hour`, `day_of_week` | ❌ Near-zero correlation — excluded from base features |
| `vote_ratio` | ❌ Dropped after redundancy check |
| `word_zscore` | ❌ Pure duplicate of `word_count` |
| SVD for HGB | ✅ Required — HGB doesn't handle sparse input natively |
