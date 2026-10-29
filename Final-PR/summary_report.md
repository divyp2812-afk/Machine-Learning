# Strategic Executive Summary Report: Telecom Customer Churn Prediction


## 1. Problem Framing & Economic Foundation
In the telecom industry (mirroring operations at Jio and Airtel), subscriber retention is 5 to 7 times more cost-effective than replacement acquisition. Customer churn directly destroys recurring revenue and depresses Customer Lifetime Value (CLV), while customer conquest inflates Customer Acquisition Cost (CAC). 

In binary churn classification, a **False Negative (missing an actual churner)** is vastly more expensive than a **False Positive (unnecessary retention perk)** ($\text{Cost}(FN) \gg \text{Cost}(FP)$). A missed churner leaves permanently for a competitor, whereas a false alarm incurs only a minor promotional discount. Therefore, **Recall on the Churn class** is established as our primary operational optimization KPI.

---

## 2. Dataset Loading & Exploratory Data Analysis (EDA)
The IBM Telco dataset comprises 7,043 subscriber records with 21 attributes across demographics, service subscriptions, contract types, and billing methods.
- **Data Hygiene:** 11 records loaded with blank whitespace strings in `TotalCharges`. Root-cause analysis confirmed these were new customers with `tenure = 0` who had not yet completed a billing cycle; these were converted to numeric and imputed with the dataset median ($1,397.48).
- **Target Distribution:** 26.54% churn (1,869 subscribers) vs 73.46% retention (5,174 subscribers), showing moderate class imbalance (~3:1).
- **Key EDA Finding:** Subscribers on Month-to-month contracts with tenure $\le 12$ months and monthly charges $\ge \$70$ exhibit an alarming **>65% churn rate** (over 2.5x the portfolio average). Additionally, `tenure` and `TotalCharges` exhibit strong collinearity ($r \approx 0.83$).

---

## 3. Data Preprocessing & Feature Engineering
To maximize model expressiveness while preventing data leakage, we implemented structured transformations:
- **Feature Engineering:** Created `tenure_group` (ordinal: New, Mid, Senior, Loyal), `num_services` (aggregate count of 6 add-on services), and `AutoPay` (binary flag for automatic bank/credit card debits).
- **Encoding & Scaling:** Binary columns were mapped to 0/1, nominal categories (`Contract`, `InternetService`, `PaymentMethod`) were one-hot encoded (`drop_first=True`), and numerical features were standardized using `StandardScaler`.
- **SMOTE Balancing:** To correct class imbalance without leaking test signals, **SMOTE** was fitted strictly on the 80% training split (balancing classes 1:1 at 4,139 samples each), keeping the 20% test set original and unaugmented.

---

## 4. Model Building I — KNN & Naive Bayes
- **K-Nearest Neighbors (KNN):** Tuned across $k \in [1, 3, 5, 7, 9, 11, 15]$ using cross-validation. Optimal $k=3$ achieved **Recall = 67.91%**, Precision = 47.65%, F1 = 0.5601, and AUC-ROC = 0.7510.
- **Gaussian Naive Bayes:** Produced strong **Recall = 77.54%**, Precision = 50.35%, F1 = 0.6105, and AUC-ROC = 0.8229. However, the foundational conditional independence assumption is heavily violated due to collinearity between `tenure`, `MonthlyCharges`, and `TotalCharges`.

---

## 5. Model Building II — SVM & Decision Tree
- **Support Vector Machine (SVM):** Tuned regularization parameter $C \in [0.1, 1, 10, 100]$ using 3-fold CV. Best $C=100$ (RBF kernel) yielded **Recall = 62.03%**, Precision = 50.99%, F1 = 0.5597, and AUC-ROC = 0.7794. While effective, SVM functions as an opaque mathematical black-box with longer training latency (6.16s).
- **Decision Tree Classifier:** Tuned `max_depth` across $[3, 4, 5, 6, 7, 8, \text{None}]$ with cost-sensitive `class_weight='balanced'`. Depth 4 proved optimal, delivering **Recall = 78.34%**, Precision = 51.13%, F1 = 0.6188, and AUC-ROC = 0.8176 in just 0.021s.
- **SMOTE vs class_weight:** Both approaches produced high churn recall (>78%), with SMOTE showing comparable performance to class-weighted loss.

---

## 6. Model Evaluation & Comparative Benchmark
All four models were evaluated on the identical 20% holdout test set (1,409 subscribers) and sorted strictly by **Recall on Churn**:

| Rank | Model Architecture | Recall (Churn KPI) | Precision | F1-Score | AUC-ROC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Decision Tree (depth=4)** | **78.34%** | **51.13%** | **0.6188** | **0.8176** | **0.021s** |
| 🥈 | **Gaussian Naive Bayes** | 77.54% | 50.35% | 0.6105 | 0.8229 | 0.009s |
| 🥉 | **K-Nearest Neighbors (k=3)** | 67.91% | 47.65% | 0.5601 | 0.7510 | 0.003s |
| 4 | **SVM (RBF Kernel, C=100)** | 62.03% | 50.99% | 0.5597 | 0.7794 | 6.165s |

**Operational Deployment Strategy (500 Calls/Day Capacity):**  
The retention call center maintains a ceiling of 500 outbound calls daily. Rather than calling randomly, subscribers are ranked by model-predicted churn probability $P(\text{Churn})$ in descending order. Selecting the **top 500 highest-probability churners daily** concentrates intervention capacity on subscribers with near-certain churn intent, maximizing strike-rate precision and saving maximum CLV.

---

## 7. Error Analysis & Top Churn Signals
- **False Negative Profiling:** Subscribers misclassified as 'Stay' ($y=1, \hat{y}=0$) had significantly higher average tenure (28.4 months vs 18.0 months for typical churners) and held 1- or 2-year contracts. They left due to unobserved operational issues (e.g., unresolved network tickets or competitor 5G promotions) rather than standard pricing friction.
- **Top 3 Churn Signals:**
  1. *Contract Commitment:* Month-to-month contracts churn at **42.7%** vs **11.3% for 1-year** and **2.8% for 2-year**.
  2. *Tenure Fragility:* **47.4%** of churn occurs within months 0–12.
  3. *Service Deficit:* Fiber optic subscribers without `TechSupport` churn at double the rate of supported users.

---

## 8. Pipeline, Production Deployment & Future Roadmap
- **Production Pipeline Serialization:** Wrapped preprocessing (`StandardScaler` + passthrough) and the Decision Tree into an end-to-end `sklearn.pipeline.Pipeline`, serialized to **`churn_model.pkl`** via `joblib`. Real-time inference was verified on 5 sample subscribers.
- **Actionable Retention Strategy:**
  - Launch a 12-month migration incentive (₹150 discount + free OTT bundle) for month-to-month users.
  - Establish a 90-day proactive onboarding care protocol for new subscribers.
  - Bundle 6 months of complimentary TechSupport with high-tier fiber plans.
- **Next Steps & Roadmap:**
  1. *Network Telemetry Ingestion:* Integrate real-time cell tower handoff failures, call-drop frequencies, and speed test logs.
  2. *Gradient Boosting Models:* Benchmark LightGBM, XGBoost, and CatBoost to push Churn Recall above 85%.
  3. *Event-Driven Scoring API:* Deploy `churn_model.pkl` as a containerized FastAPI microservice with Kafka streaming for real-time scoring when a SIM port-out or billing dispute occurs.
