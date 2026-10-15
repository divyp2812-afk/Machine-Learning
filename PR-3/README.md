
# 🛡️ PR-3 · Risk Alert Classifier

### Customer Risk Prediction Using Machine Learning

**Logistic Regression · Class Balancing · Decision Tree · Random Forest**

An end-to-end academic practical covering data preparation, imbalanced classification, model comparison, and hyperparameter tuning.

**4,600 Records** &nbsp; | &nbsp; **18 Input Features** &nbsp; | &nbsp; **Binary Classification**

</div>

---

## 📌 Project Overview

The Risk Alert Classifier predicts a customer's risk status using financial, account, and transaction-related information. The practical starts with Logistic Regression, studies how class imbalance affects predictions, and compares tree-based models before tuning a Random Forest.

The goal is to understand both **overall prediction accuracy** and **the ability to detect risky customers**. A model can achieve high accuracy while still missing important positive cases.

## 🎯 Objectives

- Identify input features and the target variable.
- Handle missing numerical and categorical values.
- Build a Logistic Regression baseline.
- Compare undersampling, oversampling, SMOTE, and ADASYN.
- Evaluate Decision Tree and Random Forest classifiers.
- Tune model parameters using cross-validation.
- Interpret confusion matrices, precision, recall, F1-score, and ROC-AUC.

## 📂 Project Files

| File | Description |
| :--- | :--- |
| `PR-3(2).ipynb` | Theory answers, preprocessing, model training, comparisons, and ROC visualization |
| `PR-3.csv` | Customer dataset used in the practical |
| `README.md` | Project overview, execution instructions, results, and observations |

Keep these files in the same folder when running the notebook.

## 📊 Dataset Summary

| Property | Details |
| :--- | :--- |
| Total records | 4,600 |
| Total columns | 19 |
| Input columns | 18 |
| Target column | `risk_status` |
| Target values | `0` = non-risk; `1` = risk, as interpreted in this practical |
| Class 0 | 4,043 records — 87.89% |
| Class 1 | 557 records — 12.11% |
| Training/testing split | 80% / 20%, stratified by target |
| Training records | 3,680 |
| Testing records | 920 |
| Random seed | `42` |

### Feature Groups

| Group | Columns |
| :--- | :--- |
| Identification | `customer_id` |
| Customer profile | `age`, `gender`, `region`, `employment_type` |
| Financial information | `annual_income_inr`, `credit_score`, `credit_utilization_ratio`, `debt_balance_inr` |
| Payment behaviour | `missed_payments_12m`, `avg_late_payment_days` |
| Transaction activity | `monthly_transaction_count`, `monthly_spend_inr`, `cash_advance_count_6m`, `last_transaction_date` |
| Account activity | `complaints_last_6m`, `failed_login_attempts_3m`, `account_tenure_months` |

### Missing Values in the Original Dataset

| Column | Missing values |
| :--- | ---: |
| `age` | 140 |
| `region` | 102 |
| `employment_type` | 144 |
| `annual_income_inr` | 166 |
| `credit_score` | 216 |
| `credit_utilization_ratio` | 147 |
| `monthly_spend_inr` | 129 |
| **Total** | **1,044** |

All remaining columns have no missing values. The notebook reports zero missing values after preprocessing.

## ⚙️ Methodology

1. **Load and inspect data:** Read the CSV using Pandas and identify features and target.
2. **Prepare data:** Apply `KNNImputer(n_neighbors=5)` to numerical columns, ordinal-encode object/category columns, and impute their missing values.
3. **Split data:** Use `train_test_split` with `test_size=0.2`, `stratify=y`, and `random_state=42`.
4. **Train the baseline:** Fit Logistic Regression and calculate classification metrics.
5. **Handle imbalance:** Resample the training set using four techniques and retrain Logistic Regression.
6. **Compare classifiers:** Train Decision Tree and Random Forest models and compare training and testing accuracy.
7. **Tune parameters:** Use `RandomizedSearchCV` and `GridSearchCV` with accuracy scoring.
8. **Evaluate results:** Compare tuned and untuned Random Forest models and plot ROC curves for the three original classifiers.

### Class Balancing Methods

| Method | How it works | Training class counts: 0 / 1 |
| :--- | :--- | :---: |
| Original training data | Retains the original imbalance | 3,234 / 446 |
| Random undersampling | Removes majority-class examples | 446 / 446 |
| Random oversampling | Repeats minority-class examples | 3,234 / 3,234 |
| SMOTE | Creates synthetic minority examples between neighbours | 3,234 / 3,234 |
| ADASYN | Creates more synthetic examples around difficult minority cases | 3,234 / 3,220 |

## 🏆 Results and Comparison

> These values are taken from the notebook's saved outputs. They describe the recorded experiment and have not been independently reproduced for this README.

### Logistic Regression: Effect of Class Balancing

Precision, recall, and F1-score below refer to the positive class (`risk_status = 1`).

| Training approach | Accuracy | Precision | Recall | F1-score |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | **98.04%** | **92.66%** | 90.99% | **91.82%** |
| Undersampling | 95.54% | 74.65% | 95.50% | 83.79% |
| Oversampling | 95.22% | 73.10% | 95.50% | 82.81% |
| SMOTE | 96.20% | 77.54% | **96.40%** | 85.94% |
| ADASYN | 95.43% | 73.79% | **96.40%** | 83.59% |

**Observation:** The baseline provides the highest precision and F1-score among these Logistic Regression experiments. SMOTE and ADASYN improve recall, reducing missed risky cases from **10 to 4**, while increasing false alerts. SMOTE produces fewer false alerts than ADASYN: **31 versus 38**.

### Baseline Confusion Matrix

| Actual / Predicted | Non-risk (0) | Risk (1) |
| :--- | ---: | ---: |
| Non-risk (0) | 801 — True Negatives | 8 — False Positives |
| Risk (1) | 10 — False Negatives | 101 — True Positives |

- **False positive:** A non-risk customer receives a risk alert.
- **False negative:** A risky customer is missed by the model.

### Classifier Performance

| Model | Training accuracy | Testing accuracy | ROC-AUC |
| :--- | ---: | ---: | ---: |
| Logistic Regression | Not reported | 98.04% | 0.992728 |
| Decision Tree | 98.21% | 97.28% | 0.916486 |
| Random Forest — untuned | 98.99% | 99.24% | **0.999800** |
| Random Forest — grid-tuned | **99.78%** | **99.57%** | Not reported |

**Best recorded testing accuracy:** The grid-tuned Random Forest achieves **99.57%**, compared with **99.24%** for the untuned Random Forest. This is an improvement of approximately **0.33 percentage points** on the same test set.

### Selected Grid Search Parameters

| Parameter | Selected value |
| :--- | :--- |
| `n_estimators` | `50` |
| `max_depth` | `10` |
| `min_samples_split` | `5` |
| `min_samples_leaf` | `1` |
| Cross-validation | 3 folds |
| Scoring metric | Accuracy |
| Best mean cross-validation accuracy | 99.05% |

The notebook also includes two randomized searches using 100 parameter combinations and 5-fold cross-validation. Both use a **Random Forest estimator** and **accuracy scoring**, even where variable names or printed labels suggest otherwise.

## 📐 Evaluation Metrics

| Metric | Meaning |
| :--- | :--- |
| Accuracy | Fraction of all predictions that are correct |
| Precision | Fraction of predicted risk cases that are actually risky: `TP / (TP + FP)` |
| Recall / TPR | Fraction of actual risk cases detected: `TP / (TP + FN)` |
| F1-score | Balance between precision and recall: `2 × Precision × Recall / (Precision + Recall)` |
| FPR | Fraction of non-risk cases incorrectly flagged: `FP / (FP + TN)` |
| ROC-AUC | Measures how well probability scores rank positive cases above negative cases across thresholds |

**Why accuracy alone is insufficient:** About 88% of this dataset belongs to class 0. Predicting every customer as non-risk would therefore achieve about 88% accuracy while detecting no risky customers.

## 💻 Tools and Libraries

| Tool / Library | Purpose |
| :--- | :--- |
| Python | Programming language |
| Jupyter Notebook | Interactive execution and documentation |
| Pandas and NumPy | Data handling and numerical operations |
| Scikit-learn | Preprocessing, classifiers, metrics, and parameter search |
| Imbalanced-learn | Undersampling, oversampling, SMOTE, and ADASYN |
| Matplotlib | ROC curve visualization |


### 4. Execute the cells in order

Use **Restart Kernel and Run All Cells** to execute the complete practical. The randomized searches may take longer because they evaluate many model configurations across multiple folds. Results may vary with library versions; the original environment's exact versions are not recorded.

## 🔍 Limitations and Future Improvements

These points explain the current implementation and how its evaluation could be strengthened:

- **Prevent preprocessing leakage:** The notebook fits imputation and encoding before the train/test split. Fit these transformations only on training data, preferably inside a pipeline that also handles cross-validation. Current scores should be treated as preliminary.
- **Resolve convergence warnings:** Saved Logistic Regression outputs contain convergence warnings. Scale numerical features using training data and increase `max_iter`, then compare the updated results. `StandardScaler` is imported but not applied in the current notebook.
- **Improve categorical handling:** Ordinal encoding introduces artificial ordering, and KNN imputation can produce fractional category codes. Consider categorical imputation and one-hot encoding, with a resampling method suitable for mixed data such as SMOTENC.
- **Review feature usefulness:** Evaluate removing `customer_id` and convert `last_transaction_date` into meaningful date features instead of treating it as an ordinal category.
- **Evaluate the selected model more fully:** Report positive-class precision, recall, F1-score, a confusion matrix, and PR-AUC for the tuned Random Forest. Its highest accuracy alone does not establish the best risk-detection performance.
- **Test generalization:** Validate on new data or an appropriate time-based holdout and select the alert threshold according to the cost of missed risks and false alerts.

## 🎓 Learning Outcomes

After completing this practical, a student can:

- Explain binary classification and Logistic Regression.
- Distinguish Type-I errors (false alerts) from Type-II errors (missed risks).
- Recognize why imbalanced datasets need more than accuracy for evaluation.
- Compare the benefits and trade-offs of four resampling techniques.
- Interpret training/testing performance and ROC curves.
- Use cross-validation to search for better model parameters.
- Identify data leakage and other limitations in a machine learning workflow.

## ✅ Conclusion

The practical demonstrates a complete customer risk classification workflow. The grid-tuned Random Forest achieves the highest recorded testing accuracy of **99.57%**, while the Logistic Regression experiments show how resampling increases recall at the cost of more false alerts. A stronger final evaluation would fit preprocessing only on training data and assess the tuned model using risk-class metrics alongside accuracy.

---

<div align="center">

**PR-3 · Risk Alert Classifier**  
*Understand the data. Compare the models. Explain the results.*

</div>
