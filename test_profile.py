import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from salary_prediction import load_dataset, clean_dataset, build_preprocessor

def run_profile():
    t0 = time.time()
    raw_df = load_dataset()
    clean_df, preprocess_info = clean_dataset(raw_df)
    model_df = clean_df.sample(n=10000, random_state=42).reset_index(drop=True)
    t1 = time.time()
    print(f"Data loading and prep: {t1-t0:.2f}s")
    
    FEATURE_COLUMNS = ["job_title", "experience_years", "education_level", "skills_count", "industry", "company_size", "location", "remote_work", "certifications"]
    X = model_df[FEATURE_COLUMNS]
    y = model_df["salary_log"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor()
    
    base_models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.0005, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=16),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=50, max_depth=18, n_jobs=1),
    }
    
    t_start = time.time()
    for model_name, estimator in base_models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        print(f"Fit {model_name}: {time.time() - t_start:.2f}s")
        t_start = time.time()
        
    tuning_grid = {
        "model__max_depth": [10, 14, 18, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    dt_tuner = GridSearchCV(
        estimator=Pipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeRegressor(random_state=42))]),
        param_grid=tuning_grid,
        scoring="r2",
        cv=3,
        n_jobs=1,
    )
    t_start = time.time()
    dt_tuner.fit(X_train, y_train)
    print(f"Fit GridSearchCV: {time.time() - t_start:.2f}s")
    
    unsupervised_df = model_df[FEATURE_COLUMNS].copy()
    unsupervised_encoded = pd.get_dummies(unsupervised_df, drop_first=True)
    unsupervised_scaled = StandardScaler().fit_transform(unsupervised_encoded)
    
    t_start = time.time()
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(unsupervised_scaled)
    print(f"Fit KMeans (7 loops): {time.time() - t_start:.2f}s")

run_profile()
