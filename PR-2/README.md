# 🏠 PR-2: Robust Regression Engine

## Machine Learning Based House Price Prediction and Model Comparison

This project builds a **Robust Regression Engine** for predicting **house prices in INR**. It compares regularized linear regression, tree-based regression, and Support Vector Regression (SVR), while also studying cross-validation, hyperparameter tuning, feature scaling, and overfitting/underfitting.

The complete implementation is available in the Jupyter Notebook **`PR-2.ipynb`** and uses the dataset **`PR-2.csv`**.

---

## 📌 Project Objective

The main objectives of this project are to:

- Prepare house-price data for machine learning.
- Identify input features and the target variable.
- Apply train-test splitting and feature scaling.
- Implement **Ridge (L2)** and **Lasso (L1)** regression.
- Tune regularization strength using cross-validation.
- Compare different cross-validation strategies.
- Implement **Decision Tree** and **Random Forest** regression.
- Implement **Linear, Polynomial, and RBF SVR**.
- Tune SVR hyperparameters.
- Evaluate all models using regression metrics.
- Detect possible overfitting and underfitting.
- Select the best-performing model from the experimental results.

---

## 📊 Dataset

The project uses house/property information to predict:

**Target Variable:** `house_price_inr`

### Features Used

| Feature | Description |
|---|---|
| `area_sqft` | Property area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location_score` | Score representing location quality |
| `property_age` | Age of the property |
| `distance_city_km` | Distance from the city |
| `near_school` | Whether a school is nearby |
| `near_metro` | Whether a metro station is nearby |
| `crime_rate_index` | Crime-rate indicator |
| `sale_year` | Year of property sale |
| `sale_month` | Month of property sale |

`property_id` and the original `sale_date` are excluded from the normal model input after the required date information is extracted.

### Data Split

- **Training samples:** 3,040
- **Testing samples:** 760
- **Train/Test ratio:** 80% / 20%
- **Random state:** 42

---

## 🛠️ Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn

Main Scikit-learn components used:

`StandardScaler`, `Ridge`, `Lasso`, `GridSearchCV`, `KFold`, `StratifiedKFold`, `LeaveOneOut`, `TimeSeriesSplit`, `DecisionTreeRegressor`, `RandomForestRegressor`, and `SVR`.

---



## 📚 Part A — Conceptual Foundation

The notebook first explains the important concepts required for the practical:

1. Regularization and why it is needed.
2. Difference between Ridge Regression (L2) and Lasso Regression (L1).
3. Cross-validation and its importance.
4. K-Fold, Stratified K-Fold, LOOCV, and Time Series Split.
5. Why tree-based models are less sensitive to feature scaling.

---

## 🧹 Part B — Dataset Preparation

The dataset is loaded using Pandas. `sale_date` is converted into a date and used to derive `sale_year` and `sale_month`.

The data is then divided into features (`X`) and target (`y`) and split into training and testing sets.

`StandardScaler` is applied where scaling is required, especially for **Ridge, Lasso, and SVR**.

---

## 📈 Part C — Regularized Linear Models

### Ridge Regression (L2)

Ridge Regression reduces large coefficients using an L2 penalty while generally retaining all features.

### Lasso Regression (L1)

Lasso Regression uses an L1 penalty and can shrink less useful feature coefficients toward zero.

### Alpha Tuning

`GridSearchCV` tests:

```text
0.01, 0.1, 1, 10, 100
```

Best values obtained in the notebook:

| Model | Best Alpha |
|---|---:|
| Ridge | **1** |
| Lasso | **100** |

The notebook also compares training error, validation error, and Ridge/Lasso coefficient behaviour.

---

## 🔁 Part D — Cross-Validation Strategies

Four validation strategies are studied:

### 1. K-Fold Cross-Validation
The training data is divided into 5 folds and each fold is used once for validation.

### 2. Stratified K-Fold
Because house price is continuous, the target is first divided into bins using `pd.qcut()`. Stratification then maintains similar price groups across folds.

### 3. Leave-One-Out Cross-Validation (LOOCV)
Each observation is used once as validation while all remaining observations are used for training.

### 4. Time Series Split
The dataset is sorted using `sale_date`, allowing earlier observations to train the model and later observations to validate it.

The notebook compares Ridge and Lasso MSE across these strategies to study model stability.

---

## 🌳 Part E — Tree-Based Regression

### Decision Tree Regression

A Decision Tree is trained as a single-tree regression model.

Tree complexity is also controlled using:

- `max_depth = 5`
- `min_samples_split = 10`
- `min_samples_leaf = 5`

### Random Forest Regression

Random Forest combines multiple decision trees. The implementation uses:

```text
n_estimators = 100
random_state = 42
```

This allows a direct comparison between a **single Decision Tree** and an **ensemble of trees**.

---

## 🤖 Part F — Support Vector Regression

Three SVR kernels are evaluated:

- **Linear Kernel**
- **Polynomial Kernel**
- **RBF Kernel**

SVR is trained using scaled features.

### SVR Hyperparameter Tuning

The RBF model is tuned using GridSearchCV with:

- `C`: 1, 10, 100
- `gamma`: scale, auto
- `epsilon`: 0.1, 0.5

Best parameters obtained:

```text
C = 100
epsilon = 0.1
gamma = scale
```

---

## 📏 Part G — Evaluation Metrics

All major models are evaluated using:

### Mean Squared Error (MSE)
Measures the average squared prediction error. **Lower is better.**

### Mean Absolute Error (MAE)
Measures the average absolute difference between actual and predicted prices. **Lower is better.**

### Root Mean Squared Error (RMSE)
Represents prediction error in the same unit as the target (INR). **Lower is better.**

### R² Score
Shows how much variation in house prices is explained by the model. **Higher is better.**

---

## 🏆 Model Performance

Results recorded in the notebook:

| Model | MSE | MAE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Ridge | 6.451 × 10¹² | ₹1,945,385 | ₹2,539,952 | 0.9199 |
| Lasso | 6.450 × 10¹² | ₹1,945,525 | ₹2,539,728 | 0.9199 |
| Decision Tree | 1.083 × 10¹³ | ₹2,395,188 | ₹3,290,412 | 0.8656 |
| **Random Forest** | **5.786 × 10¹²** | **₹1,757,755** | **₹2,405,402** | **0.9282** |
| Tuned SVR | 8.059 × 10¹³ | ₹6,985,495 | ₹8,976,954 | -0.0006 |

### ⭐ Best Performing Model

Based on the notebook's test results, **Random Forest Regression is the best-performing model** because it has:

- The **lowest MSE** among the compared models.
- The **lowest MAE**.
- The **lowest RMSE**.
- The **highest R² score (≈ 0.9282)**.

This means its predictions are, on average, closer to the actual house prices than the other tested models in this experiment.

---

## 🔍 Model Comparison

### Regularized Linear Models

**Ridge and Lasso** provide very similar performance. Regularization controls coefficient size and helps improve generalization.

### Tree-Based Models

The **Decision Tree** is simple and interpretable but can overfit. **Random Forest** combines many trees and achieves better test performance in this project.

### Support Vector Regression

Linear, Polynomial, and RBF SVR were tested. In the current experiment, SVR produced much lower R² scores than Ridge, Lasso, Decision Tree, and Random Forest.

---

## ⚠️ Overfitting and Underfitting

The project compares training and testing R² scores to identify fitting behaviour.

```text
High Train R² + Much Lower Test R² → Overfitting
Low Train R² + Low Test R²         → Underfitting
High and Similar Train/Test R²     → Good Generalization
```

Tree complexity controls and ensemble learning are used to reduce the risk of overfitting.

---

## 💡 Key Learning Outcomes

After completing this practical, we understand how to:

- Build regression models for a real-world prediction problem.
- Apply L1 and L2 regularization.
- Tune model hyperparameters.
- Use multiple cross-validation techniques.
- Handle continuous targets in Stratified K-Fold using binning.
- Respect chronological order using Time Series Split.
- Compare linear and non-linear regression models.
- Evaluate models using multiple regression metrics.
- Detect overfitting and underfitting.
- Select a model based on experimental evidence instead of only training accuracy.

---

## 💼 Business Interpretation

A house-price prediction model can help estimate property values from characteristics such as area, number of rooms, location quality, property age, distance from the city, nearby facilities, crime rate, and sale timing.

More reliable price estimates can support decisions related to **property pricing, investment analysis, market planning, and valuation**. In this experiment, Random Forest provided the strongest predictive performance among the tested models.

---

## 📝 Final Conclusion

This project demonstrates a complete regression workflow, starting from dataset preparation and feature scaling and progressing through regularization, cross-validation, tree-based learning, SVR, hyperparameter tuning, and final model evaluation.

The experimental results show that **Random Forest Regression achieved the strongest overall test performance**, with an R² score of approximately **0.9282** and an RMSE of approximately **₹2.41 million**. Ridge and Lasso also performed strongly, while the tested SVR configurations were not competitive on this dataset and setup.

The project highlights why model comparison should consider several evaluation metrics and validation strategies rather than relying on a single score.

---

### Project: PR-2 — Robust Regression Engine
### Domain: Machine Learning / Regression Analysis
### Dataset: House Price Prediction
