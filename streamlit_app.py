import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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
from pathlib import Path

st.set_page_config(page_title="Complete ML Workflow App", layout="wide")

DATA_PATH = Path(__file__).with_name("job_salary_prediction_dataset.csv")

FEATURE_COLUMNS = [
    "job_title", "experience_years", "education_level", "skills_count",
    "industry", "company_size", "location", "remote_work", "certifications",
]

TARGET_COLUMN = "salary"

@st.cache_data(show_spinner=False)
def load_dataset(sample_size=15000):
    """Load and optionally sample dataset for Streamlit Cloud compatibility"""
    try:
        df = pd.read_csv(DATA_PATH, nrows=None)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data(show_spinner=False)
def clean_dataset(df: pd.DataFrame):
    """Clean and preprocess dataset"""
    clean_df = df.copy()
    
    numeric_cols = ["experience_years", "skills_count", "certifications", "salary"]
    imputer = SimpleImputer(strategy="median")
    clean_df[numeric_cols] = imputer.fit_transform(clean_df[numeric_cols])
    
    categorical_cols = ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]
    for col in categorical_cols:
        if col in clean_df.columns:
            clean_df[col].fillna(clean_df[col].mode()[0] if not clean_df[col].mode().empty else "Unknown", inplace=True)
    
    clean_df = clean_df.drop_duplicates()
    clean_df = clean_df[(clean_df["salary"] > 0) & (clean_df["experience_years"] >= 0)]
    
    return clean_df

@st.cache_resource(show_spinner=False)
def build_preprocessor():
    """Build preprocessing pipeline"""
    return ColumnTransformer([
        ("nominal_cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), 
         ["job_title", "industry", "location"]),
        ("ordinal_cat", OrdinalEncoder(categories=[
            ["High School", "Diploma", "Bachelor", "Master", "PhD"],
            ["Startup", "Small", "Medium", "Large", "Enterprise"],
            ["No", "Hybrid", "Yes"]
        ], handle_unknown="use_encoded_value", unknown_value=-1),
         ["education_level", "company_size", "remote_work"]),
        ("numeric", StandardScaler(), ["experience_years", "skills_count", "certifications"]),
    ])

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame):
    """Prepare X, y, and train/test split"""
    X = df[FEATURE_COLUMNS].copy()
    y = np.log1p(df[TARGET_COLUMN].copy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    """Train all models"""
    preprocessor = build_preprocessor()
    results = {}
    
    # Linear Regression (Baseline)
    lr_pipe = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    results["Linear Regression"] = {
        "model": lr_pipe,
        "predictions": y_pred_lr,
        "r2": r2_score(y_test, y_pred_lr),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "mae": mean_absolute_error(y_test, y_pred_lr),
    }
    
    # Ridge & Lasso
    for name, model in [("Ridge", Ridge(alpha=10)), ("Lasso", Lasso(alpha=0.1))]:
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "model": pipe,
            "predictions": y_pred,
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }
    
    # Decision Tree
    dt_pipe = Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeRegressor(max_depth=12, random_state=42))])
    dt_pipe.fit(X_train, y_train)
    y_pred_dt = dt_pipe.predict(X_test)
    results["Decision Tree"] = {
        "model": dt_pipe,
        "predictions": y_pred_dt,
        "r2": r2_score(y_test, y_pred_dt),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        "mae": mean_absolute_error(y_test, y_pred_dt),
    }
    
    # Decision Tree Tuned (light GridSearch)
    dt_tuned = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42))
    ])
    param_grid = {"model__max_depth": [10, 14, 18]}
    grid_search = GridSearchCV(dt_tuned, param_grid, cv=2, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred_dt_tuned = grid_search.predict(X_test)
    results["Decision Tree (Tuned)"] = {
        "model": grid_search,
        "predictions": y_pred_dt_tuned,
        "r2": r2_score(y_test, y_pred_dt_tuned),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_dt_tuned)),
        "mae": mean_absolute_error(y_test, y_pred_dt_tuned),
    }
    
    # Random Forest
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=80, max_depth=14, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    results["Random Forest"] = {
        "model": rf_pipe,
        "predictions": y_pred_rf,
        "r2": r2_score(y_test, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "mae": mean_absolute_error(y_test, y_pred_rf),
    }
    
    return results, y_test

def main():
    st.title("🎓 Complete ML Workflow: Job Salary Prediction")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_dataset(sample_size=15000)
    
    if df is None or df.empty:
        st.error("Failed to load dataset")
        return
    
    st.info(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Module 1: Data & Preprocessing",
        "🎯 Module 2: Baseline Model",
        "🔧 Module 3: Optimization",
        "🌳 Module 4: Advanced Models",
        "🔮 Make Predictions",
        "📊 Plot Gallery"
    ])
    
    # Module 1: Data & Preprocessing
    with tab1:
        st.header("Module 1: Data Understanding & Preprocessing")
        
        with st.expander("📋 Dataset Overview", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
            with col2:
                st.metric("Missing Values", df.isna().sum().sum())
                st.metric("Duplicates", df.duplicated().sum())
        
        with st.expander("🧹 Data Cleaning"):
            st.write("**Cleaning Steps:**")
            st.write("✓ Imputed missing numeric values with median")
            st.write("✓ Filled categorical missing values with mode")
            st.write("✓ Removed duplicates")
            st.write("✓ Filtered invalid salary/experience values")
            
            clean_df = clean_dataset(df)
            st.write(f"After cleaning: **{len(clean_df)} rows**")
        
        with st.expander("📊 Feature Statistics"):
            st.dataframe(df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe())
        
        with st.expander("🎨 Exploratory Data Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                clean_df["salary"].hist(bins=30, ax=ax, color="skyblue", edgecolor="black")
                ax.set_xlabel("Salary")
                ax.set_ylabel("Frequency")
                ax.set_title("Salary Distribution")
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                clean_df["experience_years"].hist(bins=20, ax=ax, color="lightcoral", edgecolor="black")
                ax.set_xlabel("Experience (Years)")
                ax.set_ylabel("Frequency")
                ax.set_title("Experience Distribution")
                st.pyplot(fig)
    
    # Module 2: Baseline Model
    with tab2:
        st.header("Module 2: Baseline Model - Linear Regression")
        
        with st.spinner("Training baseline model..."):
            clean_df = clean_dataset(df)
            X_train, X_test, y_train, y_test = prepare_data(clean_df)
            models, y_test_actual = train_models(X_train, X_test, y_train, y_test)
        
        lr_result = models["Linear Regression"]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{lr_result['r2']:.4f}")
        col2.metric("RMSE", f"{lr_result['rmse']:.2f}")
        col3.metric("MAE", f"{lr_result['mae']:.2f}")
        
        st.write("**Model Performance:**")
        st.write("Linear Regression provides a baseline understanding of how salary relates to features.")
    
    # Module 3: Optimization & Unsupervised
    with tab3:
        st.header("Module 3: Model Optimization & Unsupervised Learning")
        
        with st.spinner("Training optimized models..."):
            clean_df = clean_dataset(df)
            X_train, X_test, y_train, y_test = prepare_data(clean_df)
            models, y_test_actual = train_models(X_train, X_test, y_train, y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regularization Comparison")
            comparison_data = {
                "Model": ["Linear Regression", "Ridge", "Lasso"],
                "R² Score": [models[m]["r2"] for m in ["Linear Regression", "Ridge", "Lasso"]],
                "RMSE": [models[m]["rmse"] for m in ["Linear Regression", "Ridge", "Lasso"]],
            }
            st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
        
        with col2:
            st.subheader("K-Means Clustering")
            preprocessor = build_preprocessor()
            X_transformed = preprocessor.fit_transform(X_test)
            
            inertias = []
            for k in range(2, 9):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_transformed)
                inertias.append(kmeans.inertia_)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(2, 9), inertias, marker='o', linestyle='-', color='green')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method for K-Means")
            st.pyplot(fig)
    
    # Module 4: Advanced Models
    with tab4:
        st.header("Module 4: Advanced Models & Final System")
        
        with st.spinner("Comparing all models..."):
            clean_df = clean_dataset(df)
            X_train, X_test, y_train, y_test = prepare_data(clean_df)
            models, y_test_actual = train_models(X_train, X_test, y_train, y_test)
        
        st.subheader("Model Comparison Table")
        comparison = pd.DataFrame({
            "Model": list(models.keys()),
            "R² Score": [models[m]["r2"] for m in models.keys()],
            "RMSE": [models[m]["rmse"] for m in models.keys()],
            "MAE": [models[m]["mae"] for m in models.keys()],
        }).sort_values("R² Score", ascending=False)
        
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        
        best_model = comparison.iloc[0]["Model"]
        st.success(f"🏆 **Best Model: {best_model}** with R² = {comparison.iloc[0]['R² Score']:.4f}")
    
    # Module 5: Make Predictions
    with tab5:
        st.header("🔮 Make Predictions")
        
        with st.spinner("Training best model..."):
            clean_df = clean_dataset(df)
            X_train, X_test, y_train, y_test = prepare_data(clean_df)
            models, _ = train_models(X_train, X_test, y_train, y_test)
        
        st.write("**Enter candidate details to predict salary:**")

        category_options = {
            "job_title": sorted(clean_df["job_title"].dropna().unique().tolist()),
            "education_level": sorted(clean_df["education_level"].dropna().unique().tolist()),
            "industry": sorted(clean_df["industry"].dropna().unique().tolist()),
            "company_size": sorted(clean_df["company_size"].dropna().unique().tolist()),
            "location": sorted(clean_df["location"].dropna().unique().tolist()),
            "remote_work": sorted(clean_df["remote_work"].dropna().unique().tolist()),
        }
        
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.selectbox("Job Title", category_options["job_title"])
            experience = st.slider("Experience (years)", 0, 30, 5)
            education = st.selectbox("Education Level", category_options["education_level"])
        
        with col2:
            skills = st.slider("Skills Count", 0, 20, 5)
            industry = st.selectbox("Industry", category_options["industry"])
            company_size = st.selectbox("Company Size", category_options["company_size"])
        
        location = st.selectbox("Location", category_options["location"])
        remote = st.selectbox("Remote Work", category_options["remote_work"])
        certs = st.slider("Certifications", 0, 10, 2)
        
        if st.button("Predict Salary"):
            input_data = pd.DataFrame({
                "job_title": [job_title],
                "experience_years": [experience],
                "education_level": [education],
                "skills_count": [skills],
                "industry": [industry],
                "company_size": [company_size],
                "location": [location],
                "remote_work": [remote],
                "certifications": [certs],
            })
            
            best_model_name = "Random Forest"
            best_pipe = models[best_model_name]["model"]
            pred_log = best_pipe.predict(input_data)[0]
            pred_salary = np.expm1(pred_log)
            
            st.success(f"**Predicted Salary: ${pred_salary:,.2f}**")
    
    # Module 6: Plot Gallery
    with tab6:
        st.header("📊 Plot Gallery")
        st.write("Comprehensive visualizations of the ML workflow")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            clean_df = clean_dataset(df)
            X_train, X_test, y_train, y_test = prepare_data(clean_df)
            models, _ = train_models(X_train, X_test, y_train, y_test)
            
            model_names = list(models.keys())
            r2_scores = [models[m]["r2"] for m in model_names]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            
            ax.barh(model_names, r2_scores, color=colors[:len(model_names)])
            ax.set_xlabel("R² Score")
            ax.set_title("Model Performance Comparison")
            ax.set_xlim([0, 1])
            st.pyplot(fig)
        
        with col2:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            best_rf = models["Random Forest"]["model"]
            importances = best_rf.named_steps["model"].feature_importances_
            n_features = len(FEATURE_COLUMNS)
            feature_imp = importances[-n_features:] if len(importances) > n_features else importances
            
            ax.barh(FEATURE_COLUMNS, feature_imp, color="teal")
            ax.set_xlabel("Importance")
            ax.set_title("Top Feature Importance (Random Forest)")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
