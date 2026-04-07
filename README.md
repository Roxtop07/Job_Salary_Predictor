# Job Salary Prediction — ML Project

An end-to-end Machine Learning project that predicts tech job salaries using supervised and unsupervised learning techniques.

## Problem Statement

**Regression Problem:** Predict the annual salary (in USD) of tech professionals based on features such as job title, experience, education level, skills, industry, company size, location, remote work status, and certifications.

## Dataset

- **File:** `job_salary_prediction_dataset.csv`
- **Rows:** 250,000
- **Features (9):** `job_title`, `experience_years`, `education_level`, `skills_count`, `industry`, `company_size`, `location`, `remote_work`, `certifications`
- **Target:** `salary` (continuous, USD)

## ML Workflow Covered

| Module | Topics |
|--------|--------|
| 1. Data Preprocessing | Loading, cleaning, EDA visualisations, encoding |
| 2. Baseline Model | Linear Regression, train/test split, MAE/MSE/RMSE/R² |
| 3. Optimization | Feature scaling, Ridge & Lasso regularisation, K-Means clustering, PCA |
| 4. Advanced Models | Decision Tree, Random Forest, GridSearchCV hyperparameter tuning, model comparison |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ML pipeline
python salary_prediction.py
```

All plots are automatically saved to the `plots/` directory.

## Project Structure

```
ML_Project/
├── job_salary_prediction_dataset.csv   # Dataset
├── salary_prediction.py               # Complete ML pipeline
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── plots/                             # Auto-generated visualisations
    ├── 01_salary_distribution.png
    ├── 02_salary_by_job_title.png
    ├── 03_salary_by_education.png
    ├── ...
    └── 16_model_comparison.png
```

## Models Compared

1. Linear Regression (baseline)
2. Ridge Regression (L2 regularisation)
3. Lasso Regression (L1 regularisation)
4. Decision Tree Regressor
5. Random Forest Regressor (with GridSearchCV tuning)

## Key Results

- **Best Model:** Random Forest (tuned) — highest R² score
- **Top Features:** Experience years, company size, job title, and location
- **Insights:** Enterprise companies and locations like USA/Singapore correlate with higher salaries

## Technologies Used

- Python 3
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
