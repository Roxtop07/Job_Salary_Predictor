# Explaination of the Full ML Project (Super Simple, Viva Ready)

Hi! Think of this project like teaching a smart robot to **guess salary** from job details.

We give the robot examples:
- job title
- experience
- education
- skills
- industry
- company size
- location
- remote work
- certifications

And the robot learns a pattern to predict salary.

---

## 1) What problem are we solving?

This is a **Regression** problem.
- Regression means: predict a number.
- Our number is: **annual salary in USD**.

---

## 2) Data understanding and preprocessing (Module 1)

### What we did
1. Loaded dataset from `job_salary_prediction_dataset.csv`.
2. Checked data types, shape, missing values.
3. Handled missing values:
   - numeric columns -> median
   - categorical columns -> mode (most common value)
4. Capped extreme salary outliers (winsorization idea using quantiles).
5. Applied `log1p` transform on salary for stable training.
6. Performed EDA with plots:
   - salary distribution
   - salary vs education
   - experience vs salary

### Why this matters
- Clean data = trustworthy model.
- EDA helps us understand patterns before training.

---

## 3) Baseline supervised model (Module 2)

### What we did
1. Train-test split (80-20).
2. Built preprocessing pipeline:
   - OneHotEncoder for nominal categories
   - OrdinalEncoder for ordered categories
   - StandardScaler for numeric features
3. Trained **Linear Regression** as baseline.
4. Evaluated using:
   - MAE
   - MSE
   - RMSE
   - R²

### Why baseline is important
Baseline is like your **starting score** in a game.  
Now every next model must beat it.

---

## 4) Optimization + Unsupervised learning (Module 3)

### Optimization part
- Trained **Ridge** and **Lasso** (regularized linear models).
- Compared train R² vs test R² to detect overfitting/underfitting.
- Used feature scaling inside pipeline.

### Unsupervised part
- Used **K-Means** clustering to group similar profiles.
- Used **PCA** to reduce dimensions and visualize in 2D.
- Showed elbow plot + cluster plot + PCA explained variance.

### Why this matters
- Regularization improves generalization.
- Clustering gives business insights (salary segments).
- PCA helps us visualize high-dimensional data.

---

## 5) Advanced modelling + final system (Module 4)

### What we did
- Trained **Decision Tree** and **Random Forest**.
- Applied hyperparameter tuning using **GridSearchCV** (Decision Tree tuned model).
- Compared all models in one table.
- Selected final best model by test R² and error metrics.

### Why Random Forest/Tree models help
They catch non-linear patterns that simple linear models may miss.

---

## 6) Streamlit frontend (Complete visible workflow)

In `salary_prediction.py`, the frontend has 6 sections:
1. **Module 1: Data & Preprocessing**
2. **Module 2: Baseline Model**
3. **Module 3: Optimization + Unsupervised**
4. **Module 4: Advanced Models**
5. **Final Prediction App** (live salary prediction form)
6. **All Plot Gallery** (all generated plots visible)

So your faculty can see the full end-to-end pipeline in one UI.

---

## 7) Viva one-liners (easy to say)

- **Why regression?**  
Because salary is a continuous numeric value.

- **Why train-test split?**  
To check if model performs on unseen data.

- **Why scaling?**  
So numeric features are on similar scale and optimization is stable.

- **Why one-hot encoding?**  
ML models need numbers, not raw category text.

- **Why regularization?**  
To reduce overfitting and improve generalization.

- **Why decision tree/random forest?**  
They learn non-linear patterns and feature interactions.

- **Why PCA?**  
To reduce dimensions and visualize data structure.

- **How did you pick best model?**  
By comparing R², RMSE, MAE on test set and checking fit gap.

---

## 8) How to run

```bash
pip install -r requirements.txt
streamlit run salary_prediction.py
```

---

## 9) What to tell examiner for full marks

“My project covers all four modules: data prep + EDA, baseline supervised model, optimization with regularization and unsupervised insights (K-Means/PCA), and advanced tree models with tuning and final selection. I also deployed the full workflow to Streamlit so every step is visible and explainable.”
