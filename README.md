# Heart Disease Prediction using Machine Learning

## Overview
This project applies machine learning to predict the presence of heart disease using the UCI Heart Disease dataset. As a Biomedical Engineering student, I combined domain knowledge of cardiovascular diagnostics with a complete ML pipeline — from data exploration to model evaluation — to build an interpretable, clinically-relevant prediction tool.

## Dataset
- **Source:** UCI Machine Learning Repository — Heart Disease Dataset (Cleveland database)
- **Size:** 303 patient records, 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.)
- **Target:** Presence (1) or absence (0) of heart disease, converted from the original 5-class severity scale to a binary classification problem

## Methodology
1. **Data Cleaning** — Removed 6 records with missing values in `ca` and `thal` columns
2. **Exploratory Data Analysis** — Analyzed feature distributions, class balance, and correlations with the target variable
3. **Feature Scaling** — Applied StandardScaler for the logistic regression model
4. **Model Building** — Trained and compared two classification models:
   - Logistic Regression (baseline)
   - Random Forest Classifier
5. **Evaluation** — Used accuracy, precision, recall, and F1-score, with particular attention to **recall** (minimizing missed disease cases is critical in a medical context)

## Key Findings
- **`thal`, `ca`, and `thalach`** showed the strongest correlation with heart disease presence
- Patients with heart disease tend to have a **lower maximum heart rate (thalach)**, consistent with clinical expectations
- **Cholesterol (`chol`)** showed weak linear correlation (0.08) but ranked higher in Random Forest feature importance — suggesting a possible non-linear relationship
- Feature importance from the Random Forest model closely aligned with the correlation analysis, reinforcing the reliability of these predictors

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 86.7% | - | 83.3% | - |
| Random Forest | 88.3% | 84.0% | 87.5% | 85.7% |

Random Forest outperformed Logistic Regression, particularly in recall — an important metric for minimizing missed diagnoses.

## Tools & Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run
1. Clone this repository
2. Open `heart_disease_prediction.ipynb` in Google Colab or Jupyter Notebook
3. Run all cells sequentially

## Limitations & Future Work
- Small dataset size (303 records) limits generalizability
- Future improvements could include cross-validation, hyperparameter tuning, and testing additional models (e.g., XGBoost, SVM)
- External validation on a larger, more diverse patient dataset would strengthen clinical applicability

## Author
Muhammad Hassaan — Biomedical Engineering Student
