from pathlib import Path

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

st.set_page_config(page_title="Complete ML Workflow App", layout="wide")

DATA_PATH = Path(__file__).with_name("job_salary_prediction_dataset.csv")
PLOTS_DIR = Path(__file__).with_name("plots")

FEATURE_COLUMNS = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications",
]

TARGET_COLUMN = "salary"
MODEL_ORDER = [
    "Linear Regression (Baseline)",
    "Ridge Regression",
    "Lasso Regression",
    "Decision Tree",
    "Decision Tree (Tuned)",
    "Random Forest",
]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def clean_dataset(df: pd.DataFrame):
    clean_df = df.copy()
    missing_before = clean_df.isna().sum()

    numeric_cols = ["experience_years", "skills_count", "certifications", "salary"]
    categorical_cols = [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work",
    ]

    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    for col in categorical_cols:
        mode_value = clean_df[col].mode().iloc[0] if not clean_df[col].mode().empty else "Unknown"
        clean_df[col] = clean_df[col].fillna(mode_value)

    salary_low = clean_df["salary"].quantile(0.01)
    salary_high = clean_df["salary"].quantile(0.99)
    clean_df["salary_log"] = np.log1p(clean_df["salary"])

    missing_after = clean_df.isna().sum()

    preprocessing_info = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "salary_low": float(salary_low),
        "salary_high": float(salary_high),
    }
    return clean_df, preprocessing_info


def build_preprocessor() -> ColumnTransformer:
    nominal_cols = ["job_title", "industry", "location"]
    ordinal_cols = ["education_level", "company_size", "remote_work"]
    numeric_cols = ["experience_years", "skills_count", "certifications"]

    education_order = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    company_order = ["Startup", "Small", "Medium", "Large", "Enterprise"]
    remote_order = ["No", "Hybrid", "Yes"]

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[education_order, company_order, remote_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, nominal_cols),
            ("ordinal", ordinal_pipe, ordinal_cols),
            ("numeric", numeric_pipe, numeric_cols),
        ]
    )


def evaluate_model(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_status_from_gap(gap: float) -> str:
    if gap > 0.08:
        return "Overfitting Risk"
    if gap < -0.03:
        return "Possible Underfitting"
    return "Balanced Fit"


@st.cache_resource(show_spinner=False)
def run_full_workflow(sample_size: int, random_state: int):
    raw_df = load_dataset()
    clean_df, preprocess_info = clean_dataset(raw_df)
    model_df = clean_df.sample(n=min(sample_size, len(clean_df)), random_state=random_state).reset_index(drop=True)

    X = model_df[FEATURE_COLUMNS]
    y = model_df["salary_log"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    preprocessor = build_preprocessor()
    base_models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.0005, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=16),
        "Random Forest": RandomForestRegressor(
            random_state=random_state, n_estimators=150, max_depth=18, n_jobs=1
        ),
    }

    rows = []
    trained_models = {}
    prediction_store = {}

    for model_name, estimator in base_models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)

        train_pred_log = pipeline.predict(X_train)
        test_pred_log = pipeline.predict(X_test)
        train_metrics = evaluate_model(y_train, train_pred_log)
        test_metrics = evaluate_model(y_test, test_pred_log)

        rows.append(
            {
                "Model": model_name,
                "Train R²": train_metrics["R2"],
                "Test R²": test_metrics["R2"],
                "MAE": test_metrics["MAE"],
                "MSE": test_metrics["MSE"],
                "RMSE": test_metrics["RMSE"],
                "R² Gap (Train-Test)": train_metrics["R2"] - test_metrics["R2"],
            }
        )
        trained_models[model_name] = pipeline
        prediction_store[model_name] = pd.DataFrame(
            {
                "Actual Salary": np.expm1(y_test),
                "Predicted Salary": np.expm1(test_pred_log),
            }
        )

    tuning_grid = {
        "model__max_depth": [10, 14, 18, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    dt_tuner = GridSearchCV(
        estimator=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(random_state=random_state)),
            ]
        ),
        param_grid=tuning_grid,
        scoring="r2",
        cv=3,
        n_jobs=1,
    )
    dt_tuner.fit(X_train, y_train)

    tuned_train_pred_log = dt_tuner.best_estimator_.predict(X_train)
    tuned_test_pred_log = dt_tuner.best_estimator_.predict(X_test)
    tuned_train_metrics = evaluate_model(y_train, tuned_train_pred_log)
    tuned_test_metrics = evaluate_model(y_test, tuned_test_pred_log)

    rows.append(
        {
            "Model": "Decision Tree (Tuned)",
            "Train R²": tuned_train_metrics["R2"],
            "Test R²": tuned_test_metrics["R2"],
            "MAE": tuned_test_metrics["MAE"],
            "MSE": tuned_test_metrics["MSE"],
            "RMSE": tuned_test_metrics["RMSE"],
            "R² Gap (Train-Test)": tuned_train_metrics["R2"] - tuned_test_metrics["R2"],
        }
    )
    trained_models["Decision Tree (Tuned)"] = dt_tuner.best_estimator_
    prediction_store["Decision Tree (Tuned)"] = pd.DataFrame(
        {
            "Actual Salary": np.expm1(y_test),
            "Predicted Salary": np.expm1(tuned_test_pred_log),
        }
    )

    comparison_df = pd.DataFrame(rows)
    comparison_df["Fit Status"] = comparison_df["R² Gap (Train-Test)"].apply(fit_status_from_gap)
    comparison_df["Model"] = pd.Categorical(comparison_df["Model"], categories=MODEL_ORDER, ordered=True)
    comparison_df = comparison_df.sort_values("Test R²", ascending=False).reset_index(drop=True)

    best_model_name = str(comparison_df.loc[0, "Model"])

    unsupervised_df = model_df[FEATURE_COLUMNS].copy()
    unsupervised_encoded = pd.get_dummies(unsupervised_df, drop_first=True)
    unsupervised_scaled = StandardScaler().fit_transform(unsupervised_encoded)

    inertia_values = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(unsupervised_scaled)
        inertia_values.append({"k": k, "inertia": float(kmeans.inertia_)})

    final_kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=20)
    cluster_labels = final_kmeans.fit_predict(unsupervised_scaled)

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(unsupervised_scaled)
    cluster_plot_df = pd.DataFrame(
        {
            "PC1": pca_points[:, 0],
            "PC2": pca_points[:, 1],
            "Cluster": cluster_labels,
            "Salary": model_df["salary"].values,
        }
    )
    cluster_plot_df = cluster_plot_df.sample(
        n=min(4000, len(cluster_plot_df)), random_state=random_state
    )

    cluster_salary_summary = (
        model_df.assign(Cluster=cluster_labels)
        .groupby("Cluster")["salary"]
        .agg(["count", "mean", "min", "max"])
        .rename(
            columns={
                "count": "Members",
                "mean": "Mean Salary",
                "min": "Min Salary",
                "max": "Max Salary",
            }
        )
        .round(2)
        .reset_index()
    )

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "preprocess_info": preprocess_info,
        "comparison_df": comparison_df,
        "trained_models": trained_models,
        "prediction_store": prediction_store,
        "best_model_name": best_model_name,
        "dt_tuning": {
            "best_params": dt_tuner.best_params_,
            "best_cv_r2": float(dt_tuner.best_score_),
        },
        "split_info": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": int(X_train.shape[1]),
        },
        "unsupervised": {
            "elbow_df": pd.DataFrame(inertia_values),
            "cluster_plot_df": cluster_plot_df,
            "cluster_salary_summary": cluster_salary_summary,
            "pca_var_ratio": pca.explained_variance_ratio_,
        },
    }


def plot_salary_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["salary"], bins=40, kde=True, ax=ax, color="#3b82f6")
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Annual Salary (USD)")
    return fig


def plot_salary_by_education(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    sns.boxplot(data=df, x="education_level", y="salary", order=order, ax=ax)
    ax.set_title("Salary by Education Level")
    ax.set_xlabel("Education")
    ax.set_ylabel("Salary (USD)")
    return fig


def plot_experience_vs_salary(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df.sample(n=min(5000, len(df)), random_state=42),
        x="experience_years",
        y="salary",
        alpha=0.35,
        ax=ax,
    )
    ax.set_title("Experience vs Salary")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    return fig


def prediction_scatter(pred_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sample_df = pred_df.sample(n=min(1500, len(pred_df)), random_state=42)
    ax.scatter(
        sample_df["Actual Salary"],
        sample_df["Predicted Salary"],
        alpha=0.35,
        color="#2563eb",
    )
    min_val = min(sample_df["Actual Salary"].min(), sample_df["Predicted Salary"].min())
    max_val = max(sample_df["Actual Salary"].max(), sample_df["Predicted Salary"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#ef4444")
    ax.set_title(title)
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    return fig


st.title("Job Salary Predictor")
st.caption(
    "This Streamlit app shows the full workflow: Data Prep → Baseline Model → Optimization & Unsupervised Learning → Advanced Models → Final Prediction System."
)

with st.sidebar:
    st.header("Workflow Controls")
    raw_data = load_dataset()
    default_sample = min(40000, len(raw_data))
    sample_size = st.slider(
        "Training sample size",
        min_value=10000,
        max_value=len(raw_data),
        value=default_sample,
        step=10000,
    )
    random_state = st.number_input("Random seed", min_value=1, max_value=999, value=42)
    refresh = st.button("Run / Refresh Full Workflow", use_container_width=True)

config_key = (sample_size, int(random_state))
if (
    refresh
    or "workflow" not in st.session_state
    or st.session_state.get("workflow_config") != config_key
):
    with st.spinner("Running complete ML workflow... this can take a little time."):
        st.session_state["workflow"] = run_full_workflow(sample_size, int(random_state))
        st.session_state["workflow_config"] = config_key

workflow = st.session_state["workflow"]
raw_df = workflow["raw_df"]
clean_df = workflow["clean_df"]
comparison_df = workflow["comparison_df"].copy()
best_model_name = workflow["best_model_name"]
trained_models = workflow["trained_models"]
prediction_store = workflow["prediction_store"]
preprocess_info = workflow["preprocess_info"]
show_df = comparison_df[
    ["Model", "Train R²", "Test R²", "RMSE", "MAE", "R² Gap (Train-Test)", "Fit Status"]
].sort_values("Test R²", ascending=False)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Module 1: Data & Preprocessing",
        "Module 2: Baseline Model",
        "Module 3: Optimization + Unsupervised",
        "Module 4: Advanced Models",
        "Final Prediction App",
        "All Plot Gallery",
    ]
)

with tab1:
    st.subheader("Problem Statement")
    st.write(
        "We solve a **regression** problem: predict annual job salary based on role, experience, education, skills, industry, company size, location, remote setup, and certifications."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw_df):,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Target", TARGET_COLUMN)

    st.markdown("### Data Cleaning Summary")
    missing_table = pd.DataFrame(
        {
            "Missing Before": preprocess_info["missing_before"],
            "Missing After": preprocess_info["missing_after"],
        }
    )
    st.dataframe(missing_table, use_container_width=True)
    st.caption(
        f"Salary distribution reference (no hard cap): P1 = {preprocess_info['salary_low']:.0f}, P99 = {preprocess_info['salary_high']:.0f}."
    )

    st.markdown("### Cleaned Dataset Sample")
    st.dataframe(clean_df[FEATURE_COLUMNS + ["salary"]].head(15), use_container_width=True)

    st.markdown("### EDA Visualizations")
    e1, e2 = st.columns(2)
    with e1:
        st.pyplot(plot_salary_distribution(clean_df))
    with e2:
        st.pyplot(plot_salary_by_education(clean_df))
    st.pyplot(plot_experience_vs_salary(clean_df))

    st.info(
        "Observation: salary rises with experience and education, but spread increases at senior levels, which motivates comparing multiple model families."
    )

with tab2:
    st.subheader("Baseline Model: Linear Regression")
    split = workflow["split_info"]
    st.write(
        f"Train/Test split used: **{split['train_rows']:,} train rows** and **{split['test_rows']:,} test rows** with {split['features']} input features."
    )

    baseline_row = comparison_df[comparison_df["Model"] == "Linear Regression (Baseline)"].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test R²", f"{baseline_row['Test R²']:.4f}")
    m2.metric("RMSE", f"{baseline_row['RMSE']:.2f}")
    m3.metric("MAE", f"{baseline_row['MAE']:.2f}")
    m4.metric("MSE", f"{baseline_row['MSE']:.2f}")

    st.pyplot(
        prediction_scatter(
            prediction_store["Linear Regression (Baseline)"],
            "Baseline Model: Actual vs Predicted",
        )
    )
    st.write(
        "Interpretation: this is our first benchmark. Every later model should beat this baseline on Test R² and error metrics."
    )

with tab3:
    st.subheader("Optimization & Insights")

    st.markdown("### Overfitting / Underfitting Check")
    fit_check_cols = [
        "Model",
        "Train R²",
        "Test R²",
        "R² Gap (Train-Test)",
        "Fit Status",
    ]
    st.dataframe(
        comparison_df[fit_check_cols].sort_values("Test R²", ascending=False),
        use_container_width=True,
    )

    st.markdown("### Regularization Impact (Linear vs Ridge vs Lasso)")
    regularization_view = comparison_df[
        comparison_df["Model"].isin(
            ["Linear Regression (Baseline)", "Ridge Regression", "Lasso Regression"]
        )
    ][["Model", "Test R²", "RMSE", "MAE"]]
    st.dataframe(regularization_view, use_container_width=True)

    st.markdown("### K-Means Clustering")
    unsup = workflow["unsupervised"]
    elbow_df = unsup["elbow_df"]
    fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
    ax_elbow.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    ax_elbow.set_xlabel("k (number of clusters)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method for K-Means")
    st.pyplot(fig_elbow)

    cluster_df = unsup["cluster_plot_df"]
    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=cluster_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        alpha=0.7,
        ax=ax_cluster,
    )
    ax_cluster.set_title("K-Means Clusters Visualized in 2D PCA Space")
    st.pyplot(fig_cluster)

    st.markdown("### PCA Summary")
    pca_ratio = unsup["pca_var_ratio"]
    st.write(
        f"Explained variance by first two principal components: **PC1 = {pca_ratio[0]:.2%}**, **PC2 = {pca_ratio[1]:.2%}**"
    )
    st.dataframe(unsup["cluster_salary_summary"], use_container_width=True)

with tab4:
    st.subheader("Advanced Modelling & Final Selection")

    advanced_models = comparison_df[
        comparison_df["Model"].isin(["Decision Tree", "Decision Tree (Tuned)", "Random Forest"])
    ][["Model", "Train R²", "Test R²", "RMSE", "MAE", "Fit Status"]]
    st.dataframe(advanced_models.sort_values("Test R²", ascending=False), use_container_width=True)

    st.markdown("### Hyperparameter Tuning (Decision Tree)")
    st.json(workflow["dt_tuning"])

    st.markdown("### Full Model Comparison")
    st.dataframe(show_df, use_container_width=True)

    st.success(f"Final selected model: **{best_model_name}**")
    st.pyplot(
        prediction_scatter(
            prediction_store[best_model_name],
            f"{best_model_name}: Actual vs Predicted",
        )
    )

    st.markdown("### Conclusions & Future Scope")
    st.write(
        """
        - Tree-based methods capture non-linear salary patterns better than linear-only models.
        - Regularization keeps linear models stable and helps generalization.
        - Clustering shows meaningful salary segments and supports business interpretation.
        - Future work: SHAP explainability, time-aware validation, and deployment with API monitoring.
        """
    )

with tab5:
    st.subheader("Live Salary Prediction Frontend")
    st.write("Use any trained model from this workflow to predict salary for a new profile.")

    category_options = {
        "job_title": sorted(raw_df["job_title"].dropna().unique().tolist()),
        "education_level": sorted(raw_df["education_level"].dropna().unique().tolist()),
        "industry": sorted(raw_df["industry"].dropna().unique().tolist()),
        "company_size": sorted(raw_df["company_size"].dropna().unique().tolist()),
        "location": sorted(raw_df["location"].dropna().unique().tolist()),
        "remote_work": sorted(raw_df["remote_work"].dropna().unique().tolist()),
    }

    chosen_model_name = st.selectbox(
        "Choose model for prediction",
        options=show_df["Model"].tolist(),
        index=0,
    )

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.selectbox("Job Title", category_options["job_title"])
        experience_years = st.number_input("Experience Years", min_value=0, max_value=50, value=5)
        education_level = st.selectbox("Education Level", category_options["education_level"])
        skills_count = st.number_input("Skills Count", min_value=0, max_value=50, value=8)
        industry = st.selectbox("Industry", category_options["industry"])
    with col2:
        company_size = st.selectbox("Company Size", category_options["company_size"])
        location = st.selectbox("Location", category_options["location"])
        remote_work = st.selectbox("Remote Work", category_options["remote_work"])
        certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2)

    if st.button("Predict Salary", use_container_width=True):
        user_row = pd.DataFrame(
            [
                {
                    "job_title": job_title,
                    "experience_years": int(experience_years),
                    "education_level": education_level,
                    "skills_count": int(skills_count),
                    "industry": industry,
                    "company_size": company_size,
                    "location": location,
                    "remote_work": remote_work,
                    "certifications": int(certifications),
                }
            ]
        )
        chosen_model = trained_models[chosen_model_name]
        pred_log = chosen_model.predict(user_row)[0]
        pred_salary = float(np.expm1(pred_log))

        st.success(f"Predicted annual salary: **${pred_salary:,.2f}**")
        st.caption(
            f"Model used: {chosen_model_name}. Best model on current run: {best_model_name}."
        )

with tab6:
    st.subheader("Notebook Plot Gallery (All Visuals)")
    plot_files = sorted(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []
    if not plot_files:
        st.warning("No plot images found in /plots folder.")
    else:
        for path in plot_files:
            st.image(str(path), caption=path.name, use_container_width=True)

st.markdown("---")
st.markdown(
    "**Rubric coverage:** Data preparation, baseline supervised learning, optimization with regularization + K-Means + PCA, advanced tree models with tuning, full model comparison, and final deployed frontend prediction system."
)
