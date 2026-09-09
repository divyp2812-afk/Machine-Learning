<p align="center">
  <img src="assets\project-banner.svg" alt="Machine Learning PR-1 Project Banner" width="100%">
</p>

# Machine Learning PR-1 — House Price Prediction Using Regression

## Project Overview

This project is the **Practical Examination 1 (PR-1)** for Machine Learning. It implements an end-to-end regression workflow using a **Real Estate House Price dataset containing 4,200 property records and 12 attributes**. The project covers exploratory data analysis, regression modeling, model evaluation, Gradient Descent optimization, bias-variance analysis, and practical real-estate interpretation.

The target variable is **`house_price_inr`**, and the objective is to predict house prices using property-related features.

---

## Objectives

- Understand supervised learning and regression.
- Explore and analyze the real-estate dataset.
- Implement **Simple Linear Regression**.
- Implement **Multiple Linear Regression**.
- Implement **Polynomial Regression (Degree 2)**.
- Implement **Batch, Stochastic, and Mini-Batch Gradient Descent** from scratch.
- Evaluate regression models using standard metrics.
- Compare training and testing performance.
- Analyze **bias, variance, overfitting, and underfitting**.
- Interpret the model results from a practical business perspective.

---

## Dataset

The project uses a real-estate dataset with **4,200 records and 12 attributes** and no missing values.

### Main Features

| Feature | Description |
|---|---|
| `house_id` | Unique property identifier |
| `area_sqft` | Built-up area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location_score` | Neighborhood quality/accessibility score |
| `age_years` | Property age in years |
| `distance_city_km` | Distance from the city center |
| `lot_size_sqft` | Plot/lot size |
| `has_garage` | Garage availability |
| `has_pool` | Swimming pool availability |
| `renovation_years_ago` | Years since last renovation |
| `house_price_inr` | **Target variable — house price in INR** |

The dataset is divided into **80% training data and 20% testing data**.

---

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Models Implemented

### 1. Simple Linear Regression

Simple Linear Regression uses **`area_sqft`** as the single independent variable to predict house price.

The fitted model achieved:

- **R² Score:** 0.5625
- **RMSE:** ₹8,184,696.70

This indicates that house area alone explains about **56.25% of the variation in house prices**.

---

### 2. Multiple Linear Regression

Multiple Linear Regression uses 10 property-related features:

- `area_sqft`
- `bedrooms`
- `bathrooms`
- `location_score`
- `age_years`
- `distance_city_km`
- `lot_size_sqft`
- `has_garage`
- `has_pool`
- `renovation_years_ago`

Performance:

- **MSE:** 1.2593 × 10¹³
- **MAE:** ₹2,604,991.41
- **RMSE:** ₹3,548,650.29
- **R² Score:** 0.9178
- **Adjusted R²:** 0.9168

This was the **best-performing regression model** in the project.

---

### 3. Polynomial Regression

A **Degree-2 Polynomial Regression** model was implemented using `area_sqft`.

Performance:

- **R² Score:** 0.5627
- **RMSE:** ₹8,183,089.09

Polynomial Regression provided only a very small improvement over Simple Linear Regression, showing that adding more relevant property features is more useful than adding polynomial powers of house area alone.

---

## Model Performance Comparison

| Model | R² Score | RMSE |
|---|---:|---:|
| Simple Linear Regression | 0.5625 | ₹8,184,696.70 |
| Polynomial Regression (Degree 2) | 0.5627 | ₹8,183,089.09 |
| **Multiple Linear Regression** | **0.9178** | **₹3,548,650.29** |


<p align="center">
  <img src="assets\model-performance.svg" alt="Model Performance Comparison" width="100%">
</p>

### Best Model

**Multiple Linear Regression** is the best-performing model because it uses multiple structural, geographic, and property-related features. It explains approximately **91.78% of the variation in house prices** and produces much lower prediction error than the single-feature models.

---

## Gradient Descent Optimization

Three Gradient Descent methods were implemented from scratch using NumPy and standardized features.

### Batch Gradient Descent

- Learning Rate: `0.01`
- Epochs: `1000`
- Smooth and deterministic convergence

### Stochastic Gradient Descent

- Learning Rate: `0.001`
- Epochs: `50`
- Updates parameters after each individual sample
- Faster updates but noisier convergence

### Mini-Batch Gradient Descent

- Learning Rate: `0.01`
- Epochs: `100`
- Batch Size: `32`
- Provides a balance between speed and stable convergence

Mini-Batch Gradient Descent offered a useful balance between computational efficiency and smooth optimization.

---

## Evaluation Metrics

The following metrics are used to evaluate the models:

- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the same unit as house price.
- **R² Score:** Measures how much variation in the target is explained by the model.
- **Adjusted R²:** Adjusts R² according to the number of predictors.

---

## Bias-Variance Analysis

| Model | Train R² | Test R² | Diagnosis |
|---|---:|---:|---|
| Simple Linear Regression | 0.5723 | 0.5625 | High Bias / Underfitting |
| Multiple Linear Regression | **0.9247** | **0.9178** | **Best Balanced Fit** |
| Polynomial Regression | 0.5725 | 0.5627 | High Bias / Feature-limited |

The Multiple Linear Regression model has a generalization gap of only **0.0070**, showing that it performs similarly on both training and unseen testing data.

---

## Key Findings

- Area alone explains about **56%** of house-price variation.
- Using multiple relevant property features increases the explained variation to **91.78%**.
- Multiple Linear Regression produces the lowest prediction error.
- Degree-2 Polynomial Regression adds almost no improvement over Simple Linear Regression.
- Multiple Linear Regression shows a strong balance between bias and variance.
- Gradient Descent demonstrates how model parameters are iteratively optimized by minimizing the cost function.

---

## Practical Business Interpretation

The model can be useful for:

- Real-estate price estimation
- Automated property valuation
- Buyer and seller decision support
- Property investment analysis
- Understanding the effect of location, area, amenities, age, and renovation on property value

Positive property characteristics such as larger area, higher location score, garage availability, and pool availability can increase predicted property value, while factors such as greater property age and greater distance from the city center may reduce it.

---

## Final Conclusion

This project demonstrates a complete machine learning regression workflow for real-estate house-price prediction. It covers data understanding, exploratory analysis, Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, Gradient Descent optimization, model evaluation, and bias-variance diagnostics.

Among the tested models, **Multiple Linear Regression performed best**, achieving a **test R² of 0.9178 (91.78%)** with an **RMSE of ₹3,548,650.29**. Its small train-test performance gap also indicates good generalization without significant overfitting.
