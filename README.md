# Job Salary Prediction — End-to-End ML Project

An end-to-end Machine Learning project demonstrating the complete ML workflow: data preparation, supervised learning, model optimization, unsupervised learning, and advanced modeling techniques.

---

## 📋 Problem Statement

**Objective:** Build a regression model to predict the annual salary (in USD) of tech professionals based on job characteristics.

**Business Context:**
- Helps job seekers negotiate fair compensation
- Assists HR departments in setting competitive salary ranges
- Enables companies to benchmark against industry standards

**Target Variable:** `salary` (continuous, annual salary in USD)

**Features (9):**
| Feature | Description | Type |
|---------|-------------|------|
| `job_title` | Role/position (12 categories) | Categorical |
| `experience_years` | Years of professional experience | Numerical |
| `education_level` | Highest education attained | Ordinal |
| `skills_count` | Number of relevant technical skills | Numerical |
| `industry` | Sector of employment (10 categories) | Categorical |
| `company_size` | Size of employer organization | Ordinal |
| `location` | Geographic location (10 categories) | Categorical |
| `remote_work` | Remote work arrangement | Categorical |
| `certifications` | Number of professional certifications | Numerical |

---

## 📊 Dataset

- **File:** `job_salary_prediction_dataset.csv`
- **Rows:** 250,000
- **Columns:** 10 (9 features + 1 target)
- **Source:** Synthetic dataset for ML education

---

## 🔄 ML Workflow (Course Modules)

| Module | Topics Covered | Key Techniques |
|--------|---------------|----------------|
| **1. Data Preparation** | Loading, EDA, cleaning, encoding | Winsorization, log transform, visualizations |
| **2. Baseline Model** | Train/test split, Linear Regression | MAE, MSE, RMSE, R² evaluation |
| **3. Optimization** | Feature scaling, regularization, unsupervised learning | Ridge, Lasso, K-Means, PCA |
| **4. Advanced Models** | Tree-based models, hyperparameter tuning | Decision Tree, Random Forest, GridSearchCV |

---

## 🚀 How to Run

### Option 1: Run the Jupyter Notebook (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter and open the notebook
jupyter notebook job_salary_prediction_complete.ipynb
```

### Option 2: Run the Streamlit Web App
```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install streamlit

# 2. Launch the app
streamlit run salary_prediction.py
```

---

## 📁 Project Structure

```
Job_Salary_Predictor/
├── job_salary_prediction_complete.ipynb  # Complete ML notebook (all 4 modules)
├── job_salary_prediction_dataset.csv     # Dataset (250K records)
├── salary_prediction.py                  # Streamlit web app for predictions
├── job_salary_prediction_documentation.md # Code documentation
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── plots/                                # Auto-generated visualizations
│   ├── 01_salary_distribution.png
│   ├── 02_numerical_distributions.png
│   ├── ...
│   └── 19_best_model_predictions.png
├── linear_regression_pipeline.pkl        # Saved model
├── decision_tree_pipeline.pkl            # Saved model
└── random_forest_pipeline.pkl            # Saved model (best)
```

---

## 🤖 Models Compared

| Model | Description | Regularization |
|-------|-------------|----------------|
| Linear Regression | Baseline model | None |
| Ridge Regression | L2 regularization | Shrinks coefficients |
| Lasso Regression | L1 regularization | Feature selection |
| Decision Tree | Tree-based, tuned | Pruning (max_depth, min_samples) |
| Random Forest | Ensemble of trees, tuned | GridSearchCV optimization |

---

## 📈 Key Results

### Model Performance Comparison

| Model | R² Score | RMSE | Notes |
|-------|----------|------|-------|
| Linear Regression | ~0.85 | ~0.12 | Good baseline |
| Ridge Regression | ~0.85 | ~0.12 | Similar to baseline |
| Lasso Regression | ~0.85 | ~0.12 | Similar to baseline |
| Decision Tree | ~0.87 | ~0.11 | Requires pruning |
| **Random Forest** | **~0.90** | **~0.10** | **Best model** |

### Top Predictive Features
1. **Experience Years** — strongest correlation with salary
2. **Company Size** — Enterprise > Large > Medium > Small > Startup
3. **Location** — USA and Singapore highest paying
4. **Job Title** — ML Engineer and AI Engineer highest paid
5. **Education Level** — PhD > Master > Bachelor

### Unsupervised Learning Insights
- **K-Means Clustering** identified 4 distinct salary segments
- **PCA Analysis** revealed the feature space has high intrinsic dimensionality

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data Manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Web App | Streamlit |
| Serialization | joblib |

---

## 📝 Course Requirements Covered

✅ **Module 1:** Clear problem definition, EDA with visualizations, data cleaning  
✅ **Module 2:** Linear Regression baseline, train/test split, proper metrics  
✅ **Module 3:** Feature scaling, Ridge/Lasso regularization, K-Means, PCA  
✅ **Module 4:** Decision Tree, Random Forest, GridSearchCV, model comparison  

---

## 🔮 Future Scope

1. **Feature Engineering** — Interaction features, external economic data
2. **Advanced Models** — XGBoost, LightGBM, Neural Networks
3. **Deployment** — REST API, cloud hosting
4. **Continuous Learning** — Model retraining, drift monitoring

---

## 👤 Author

**Manish Srivastav**  
ML Course Project — Job Salary Prediction

---

## 📄 License

This project is for educational purposes as part of an ML course curriculum.
