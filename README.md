# Heart Disease Prediction using Machine Learning

## Overview
This project uses machine learning to predict whether a patient has heart disease, based on the UCI Heart Disease dataset. I'm a Biomedical Engineering student, so I tried to approach this not just as a coding exercise but by thinking about which features actually make clinical sense, and checking whether the model's findings lined up with that.

## Dataset
- **Source:** UCI Machine Learning Repository, Heart Disease Dataset (Cleveland database)
- **Size:** 303 patient records, 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.)
- **Target:** Originally a 5-class severity scale, converted to binary (1 = disease present, 0 = no disease) to keep the problem simple for a first project

## Methodology
1. **Data Cleaning:** Dropped 6 rows with missing values in the `ca` and `thal` columns
2. **Exploratory Data Analysis:** Looked at feature distributions, class balance, and correlation with the target
3. **Feature Scaling:** Applied StandardScaler before training logistic regression
4. **Models:** Trained and compared two classifiers, Logistic Regression (baseline) and Random Forest
5. **Evaluation:** Accuracy, precision, recall, and F1-score. Recall mattered most here since missing an actual disease case is worse than a false alarm in a medical setting

## Key Findings
- `thal`, `ca`, and `thalach` had the strongest correlation with heart disease
- Patients with heart disease tended to have a lower max heart rate (`thalach`), which matches what you'd expect clinically
- `chol` had weak linear correlation (0.08) but showed up higher in the Random Forest's feature importance, so the relationship might be non-linear rather than absent
- Feature importance from Random Forest mostly matched the correlation results, which was a good sanity check

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 86.7% | - | 83.3% | - |
| Random Forest | 88.3% | 84.0% | 87.5% | 85.7% |

Random Forest did better overall, especially on recall.

## Tools & Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run
1. Clone this repository
2. Open `heart_disease_prediction.ipynb` in Google Colab or Jupyter Notebook
3. Run all cells in order

## Limitations & Future Work
- Only 303 records, so results might not generalize well to a larger population
- Would like to try cross-validation, hyperparameter tuning, and maybe XGBoost or SVM
- Testing on a bigger, more diverse dataset would be needed before this could actually be useful clinically

## Author
Muhammad Hassaan, Biomedical Engineering Student
