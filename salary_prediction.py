import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="Job Salary Predictor", layout="centered", page_icon="💰")

DATA_PATH = Path(__file__).parent / "job_salary_prediction_dataset.csv"

# Map model names to their pickle files
MODEL_FILES = {
    "Random Forest": "random_forest_pipeline.pkl",
    "Decision Tree": "decision_tree_pipeline.pkl",
    "Linear Regression": "linear_regression_pipeline.pkl"
}

@st.cache_data
def get_options():
    try:
        df = pd.read_csv(DATA_PATH, usecols=[
            "job_title", "experience_years", "education_level", "skills_count",
            "industry", "company_size", "location", "remote_work", "certifications"
        ])
        return {
            "job_titles": sorted(df["job_title"].dropna().unique().tolist()),
            "education_levels": sorted(df["education_level"].dropna().unique().tolist()),
            "industries": sorted(df["industry"].dropna().unique().tolist()),
            "company_sizes": sorted(df["company_size"].dropna().unique().tolist()),
            "locations": sorted(df["location"].dropna().unique().tolist()),
            "remote_options": sorted(df["remote_work"].dropna().unique().tolist()),
        }
    except Exception as e:
        st.error(f"❌ Cannot load dataset options: {e}")
        return None

@st.cache_resource
def load_model(model_name):
    try:
        model_path = Path(__file__).parent / MODEL_FILES[model_name]
        model = joblib.load(model_path)
        # Ensure it doesn't crash on Streamlit Cloud by forcing single-threaded prediction
        if hasattr(model.named_steps.get('model'), 'n_jobs'):
            model.named_steps['model'].n_jobs = 1
        return model
    except Exception as e:
        st.error(f"❌ Cannot load {model_name} model: {e}")
        return None

def main():
    st.title("Job Salary Predictor")
    st.markdown("Enter your professional details below to predict your annual salary.")
    
    options = get_options()
    
    if not options:
        st.warning("Please ensure the dataset file is present.")
        return
        
    selected_model_name = st.selectbox("🤖 Select Prediction Model", list(MODEL_FILES.keys()))
    model = load_model(selected_model_name)
    
    if not model:
        return

    st.subheader("Your Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        job_title = st.selectbox("Job Title", options["job_titles"])
        experience_years = st.number_input("Experience (years)", min_value=0, max_value=50, value=5)
        education_level = st.selectbox("Education Level", options["education_levels"])
        skills_count = st.number_input("Skills Count", min_value=0, max_value=50, value=8)
        
    with col2:
        industry = st.selectbox("Industry", options["industries"])
        company_size = st.selectbox("Company Size", options["company_sizes"])
        location = st.selectbox("Location", options["locations"])
        remote_work = st.selectbox("Remote Work", options["remote_options"])
        
    certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2)
    
    # We use a button so the UI doesn't flicker on every single change, 
    # but without `st.form` they can freely predict multiple times without issues.
    if st.button("Predict Salary", use_container_width=True, type="primary"):
        input_data = pd.DataFrame([{
            "job_title": job_title,
            "experience_years": experience_years,
            "education_level": education_level,
            "skills_count": skills_count,
            "industry": industry,
            "company_size": company_size,
            "location": location,
            "remote_work": remote_work,
            "certifications": certifications
        }])
        
        with st.spinner(f"Calculating with {selected_model_name}..."):
            try:
                pred_log = model.predict(input_data)[0]
                prediction = np.expm1(pred_log)
                
                st.success(f"### Predicted Annual Salary: **${prediction:,.0f}**")
                st.caption(f"Predicted using {selected_model_name}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
