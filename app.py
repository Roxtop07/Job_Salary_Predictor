import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="Job Salary Predictor", layout="centered", page_icon="💰")

DATA_PATH = Path(__file__).parent / "job_salary_prediction_dataset.csv"
MODEL_PATH = Path(__file__).parent / "random_forest_pipeline.pkl"

@st.cache_data
def get_options():
    try:
        # Just read the columns we need to get unique categories
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
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        # Ensure it doesn't crash on Streamlit Cloud by forcing single-threaded prediction
        if hasattr(model.named_steps.get('model'), 'n_jobs'):
            model.named_steps['model'].n_jobs = 1
        return model
    except Exception as e:
        st.error(f"❌ Cannot load model: {e}")
        return None

def main():
    st.title("Job Salary Predictor")
    st.markdown("Enter your professional details below to predict your annual salary using a trained Random Forest model.")
    
    options = get_options()
    model = load_model()
    
    if not options or not model:
        st.warning("Please ensure the dataset and model files are present.")
        return
        
    with st.form("prediction_form"):
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
        
        submit_button = st.form_submit_button("Predict Salary", use_container_width=True)
        
    if submit_button:
        # Create a dataframe for the input
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
        
        with st.spinner("Calculating..."):
            try:
                pred_log = model.predict(input_data)[0]
                prediction = np.expm1(pred_log)
                
                st.success(f"### Predicted Annual Salary: **${prediction:,.0f}**")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
