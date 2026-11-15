# MSCS 634 Lab 4: Regression Analysis with Regularization

**Todd Simonson**
MSCS-634: Advanced Big Data and Data Mining
Dr. Satish Penmatsa

## Purpose

This lab compares regression techniques on the Diabetes dataset from scikit-learn, which covers 442 patients, 10 baseline health measurements, and a measure of disease progression one year later. It builds up in five steps:

1. Simple linear regression on one feature
2. Multiple regression on all features
3. Polynomial regression, to show underfitting and overfitting
4. Ridge and Lasso regularization, to control overfitting
5. A comparison of all models

Every model is evaluated with MAE, MSE, RMSE, and R² on the same held-out test set.

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_4.ipynb` | Notebook with data exploration, all five model types, plots, and analysis |
| `requirements.txt` | Python packages needed to run the notebook |

The Diabetes dataset ships with scikit-learn (`sklearn.datasets.load_diabetes`), so no data file is needed.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook MSCS_634_Lab_4.ipynb
```

## Results

The data was split 80/20 (353 training patients, 89 test patients) with `random_state=42`.

| Model | MAE | MSE | RMSE | Test R² | Train R² |
|---|---|---|---|---|---|
| Simple linear (BMI) | 52.26 | 4061.8 | 63.73 | 0.2334 | 0.3657 |
| Multiple linear (10 features) | 42.79 | 2900.2 | 53.85 | 0.4526 | 0.5279 |
| Polynomial degree 2 (65 features) | 43.58 | 3096.0 | 55.64 | 0.4156 | 0.6062 |
| Ridge (degree 2, alpha = 79.4 from CV) | 41.14 | 2632.9 | 51.31 | 0.5031 | 0.5661 |
| Lasso (degree 2, alpha = 3.1 from CV, 15 of 65 terms kept) | 41.23 | 2566.0 | 50.66 | **0.5157** | 0.5302 |

## Key Insights

- **More features helped, up to a point.** BMI alone explained 23% of the variation in disease progression. All 10 features explained 45%.
- **Polynomial terms overfit quickly.** Higher powers of BMI added nothing through degree 6; at degree 12, test R² fell to −1.09 while training R² kept rising. With all features, degree 2 (65 terms) already scored lower on the test set than plain multiple regression. Degree 3 (285 terms for 353 patients) reached a training R² of 0.88 and a test R² of −13.69.
- **Regularization fixed the overfitting.** Ridge and Lasso raised the degree-2 model's test R² from 0.416 to 0.503 and 0.516. Lasso cut the train-test gap from 0.19 to 0.01.
- **Alpha controls the bias-variance trade-off.** Small alpha leaves the model overfitting; large alpha underfits. Ridge shrinks all 65 coefficients gradually and peaks near alpha = 30. Lasso removes terms entirely, going from 65 to 15 to 0 as alpha grows, and peaks near alpha = 1 to 3.
- **Lasso selected the same features the data pointed to.** Of 65 terms, it kept mainly BMI, log triglycerides (s5), blood pressure, and HDL cholesterol (s3, protective), which are the features with the strongest correlations to progression.
- **Cross-validation tempers the ranking.** Across 5 folds on all 442 patients, Lasso (0.486), multiple regression (0.479), and Ridge (0.470) are statistically tied, while unregularized degree 2 trails at 0.366. The relationships in this dataset are close to linear. Regularization's main value here was preventing overfitting, not discovering new structure.
- **Dataset limits.** No model explains much more than half the variation, and all of them pull extreme cases toward the average. Disease progression clearly depends on factors that these 10 baseline measurements do not capture.

## Challenges and Decisions

1. **High polynomial degrees stopped changing.** scikit-learn ships these features already scaled to small values (roughly −0.1 to 0.2), so high powers of BMI shrink toward zero, and the fit froze from degree 8 onward for numerical rather than statistical reasons. Standardizing BMI before expanding it fixed this, so the over- and underfitting demonstration reflects real model behavior.
2. **Scaling only where it matters.** The features are already centered and scaled, so no extra scaling was applied to the simple and multiple regressions. Standardization was added inside pipelines after polynomial expansion and before Ridge and Lasso, because their penalties depend on feature scale.
3. **Choosing alpha without leaking the test set.** Choosing alpha by test-set score would tune the model to the test data. Alpha was instead chosen by 5-fold cross-validation on the training set (`RidgeCV`, `LassoCV`), and the test set was used once for final scoring. Alpha sweeps are shown separately to illustrate its effect.
4. **Where to apply regularization.** On the 10-feature linear model, Ridge and Lasso barely changed the results (test R² 0.453 to at most 0.467), because that model was not overfitting. They were applied to the 65-term polynomial model, where they had a real effect.
5. **Reading the coefficients.** Total cholesterol (s1) and LDL (s2) correlate at 0.897, which gave them large coefficients of opposite sign in the multiple regression (−931.5 and +518.1). The notebook treats this as multicollinearity rather than as evidence that one is protective.
6. **A single test split.** With 89 test patients, small differences between models can come from the particular split. 5-fold cross-validation was added in Step 6 to check which conclusions hold up.
