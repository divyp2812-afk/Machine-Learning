<div align="center">

# ✉️ Message Intelligence System
### PR-4 · Spam Classification with Machine Learning

**Three classifiers. One dataset. A practical comparison.**

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Jupyter Notebook`

**Author: Divy Patel**

</div>

---

## 📌 Project Overview

This project classifies messages using **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Gaussian Naive Bayes**. It compares distance-based, margin-based, and probabilistic approaches using the same training and testing split.

The notebook also explores conditional probability, Bayes’ theorem, support vectors, and the effect of changing KNN distance metrics and SVM kernels.

> **At a glance:** 5,200 messages · 12 numeric model features · 3 classifier families · 80/20 train–test split

## 📚 Contents

- [Dataset](#-dataset)
- [Experiments](#-experiments)
- [Results](#-results)
- [Run the Notebook](#-run-the-notebook)
- [Probability Demonstration](#-probability-demonstration)
- [Key Learnings](#-key-learnings)

## 🗂️ Dataset

| Property | Value |
|---|---|
| Uploaded dataset | `PR-4(1).csv` |
| Filename expected by the notebook | `PR-4.csv` |
| Rows | 5,200 |
| Columns | 16 |
| Target | `spam_label` |
| Label distribution | Class 0: 4,227 · Class 1: 973 |
| Model input | 12 numeric features |
| Training samples | 4,160 |
| Testing samples | 1,040 |

### Features used for classification

| Group | Features |
|---|---|
| Message structure | `message_length`, `word_count`, `num_urls`, `num_digits`, `num_special_chars` |
| Keyword indicators | `spam_keyword_score`, `legit_keyword_score` |
| Sender information | `sender_activity_score`, `sender_account_age_days`, `messages_sent_last_24h` |
| Time information | `hour_of_day`, `day_of_week` |

`message_id`, `message_text`, and `timestamp` are excluded from model input. The target `spam_label` is stored separately. Classification uses the existing numeric features; the notebook does not vectorize raw message text.

### Missing values

| Column | Missing entries |
|---|---:|
| `sender_activity_score` | 106 |
| `sender_account_age_days` | 113 |
| `messages_sent_last_24h` | 162 |

The notebook fills missing numeric values with training-data medians using `SimpleImputer` and uses `StandardScaler` for feature scaling.

## 🧪 Experiments

| Area | Work completed |
|---|---|
| Probability foundations | Conditional probability, Bayes’ theorem, and conditional independence |
| Data preparation | Numeric feature selection, standard scaling, median imputation, stratified split |
| KNN | Compare K = 1, 3, 5, 7, and 9 |
| Distance metrics | Euclidean, Manhattan, and Chebyshev |
| SVM | Linear, RBF, and degree-3 polynomial kernels |
| Margin analysis | Inspect support vectors and calculate linear SVM margin width |
| Naive Bayes | Train a GaussianNB classifier |
| Manual calculation | Compute Gaussian feature densities and normalized class probabilities |
| Verification | Compare manual probabilities with `predict_proba()` |
| Evaluation | Accuracy, precision, recall, and F1 score |

The split uses `test_size=0.2`, `random_state=42`, and `stratify=y` to preserve class proportions.

## 📊 Results

The following values come from the **saved notebook outputs**; they are not a new independent evaluation.

### Final model comparison

| Model | Configuration | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---|---:|---:|---:|---:|
| KNN | K = 5, default Euclidean distance | 100% | 1.00 | 1.00 | 1.00 |
| SVM | RBF kernel | 100% | 1.00 | 1.00 | 1.00 |
| Naive Bayes | GaussianNB | 100% | 1.00 | 1.00 | 1.00 |

**High precision and high recall:** All three final models tie in the saved results. The notebook’s `idxmax()` prints KNN because it selects the first row when scores tie; this does not establish KNN as a unique winner.

### KNN experiments

| K | Accuracy |
|---:|---:|
| 1 | 100% |
| 3 | 100% |
| 5 | 100% |
| 7 | 99.90% |
| 9 | 99.90% |

| Distance metric | Accuracy at K = 5 |
|---|---:|
| Euclidean | 100% |
| Manhattan | 100% |
| Chebyshev | 99.81% |

### SVM experiments

| Kernel | Accuracy |
|---|---:|
| Linear | 100% |
| RBF | 100% |
| Polynomial, degree 3 | 100% |

The polynomial SVM reports **67 support vectors**: 43 for class 0 and 24 for class 1. A separately trained linear SVM reports a full margin width of approximately **1.8954** in scaled feature space.

### Evaluation notes

- The current notebook scales the entire dataset **before** splitting. For a clean evaluation, split the raw features first, fit the imputer and scaler on training data, and only transform the test data.
- Select K, distance metrics, and kernels using validation data or cross-validation, then evaluate the chosen models once on an untouched test set.
- The reported perfect scores describe this dataset and workflow; they do not establish performance on new real-world messages.
- Weighted scores summarize both classes. For spam detection, inspect the spam class’s own precision and recall as well.


## 🧮 Probability Demonstration

Bayes’ theorem combines a class prior with the likelihood of observed features:

```text
Posterior probability = (Likelihood × Prior) / Evidence
```

Gaussian Naive Bayes models each numeric feature with a Gaussian distribution and assumes features are conditionally independent given the class. The notebook multiplies feature densities by each class prior, then normalizes the resulting scores.

For the first test sample, the saved manual and model probabilities both display `[1.0, 0.0]`, with predicted class `0`. The comparison using `np.allclose()` returns `True`.

For larger calculations, log probabilities help avoid numerical underflow when multiplying many small densities.

## 💡 Key Learnings

| Model | Prediction approach | Main consideration |
|---|---|---|
| KNN | Majority class among nearby training samples | Sensitive to feature scales and distance choices |
| SVM | Separating boundary with a wide margin | Kernel and parameter choices affect performance |
| Gaussian Naive Bayes | Class probabilities from Gaussian feature models | Relies on conditional independence and distribution assumptions |

- **Accuracy** measures the fraction of correct predictions.
- **Precision** measures how often predictions for a class are correct.
- **Recall** measures how many actual examples of a class are detected.
- **F1 score** balances precision and recall.

High spam precision reduces legitimate messages incorrectly flagged as spam. High spam recall reduces missed spam messages.

---

<div align="center">

**An academic exploration of message classification, probability, and model evaluation.**

</div>
