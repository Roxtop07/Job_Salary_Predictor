import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. SET UP PAGE ---
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction App")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    return {
        "Linear Regression": joblib.load('linear_regression_pipeline.pkl'),
        "Decision Tree": joblib.load('decision_tree_pipeline.pkl'),
        "Random Forest": joblib.load('random_forest_pipeline.pkl')
    }

try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Check if .pkl files are in the same folder.")
    st.stop()

# Sidebar for model selection
selected_model_name = st.sidebar.selectbox("Choose ML Model", list(models.keys()))
model = models[selected_model_name]

# --- 3. UI INPUTS (EXACT OPTIONS FROM YOUR CSV) ---
st.subheader("Enter Job Details")
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title", [
        'AI Engineer', 'Backend Developer', 'Business Analyst', 'Cloud Engineer', 
        'Cybersecurity Analyst', 'Data Analyst', 'Data Scientist', 'DevOps Engineer', 
        'Frontend Developer', 'Machine Learning Engineer', 'Product Manager', 'Software Engineer'
    ])
    experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    education_level = st.selectbox("Education Level", ['Bachelor', 'Diploma', 'High School', 'Master', 'PhD'])
    skills_count = st.number_input("Number of Relevant Skills", min_value=0, max_value=50, value=5)
    industry = st.selectbox("Industry", [
        'Consulting', 'Education', 'Finance', 'Government', 'Healthcare', 
        'Manufacturing', 'Media', 'Retail', 'Technology', 'Telecom'
    ])

with col2:
    company_size = st.selectbox("Company Size", ['Enterprise', 'Large', 'Medium', 'Small', 'Startup'])
    location = st.selectbox("Location", [
        'Australia', 'Canada', 'Germany', 'India', 'Netherlands', 
        'Remote', 'Singapore', 'Sweden', 'UK', 'USA'
    ])
    remote_work = st.selectbox("Remote Work", ['Hybrid', 'No', 'Yes'])
    certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=1)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Salary", use_container_width=True):
    
    # Create the dictionary with EXACT column order as the CSV (X = df.drop('salary', axis=1))
    input_dict = {
        'job_title': job_title,
        'experience_years': int(experience_years),
        'education_level': education_level,
        'skills_count': int(skills_count),
        'industry': industry,
        'company_size': company_size,
        'location': location,
        'remote_work': remote_work,
        'certifications': int(certifications)
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    try:
        # Step A: Get the LOG prediction from the model
        log_prediction = model.predict(input_df)
        
        # Step B: CONVERT LOG BACK TO DOLLARS (Inverse of log1p)
        actual_salary = np.expm1(log_prediction[0])
        
        # Step C: Show result
        st.balloons()
        st.success(f"### Predicted Annual Salary: ${actual_salary:,.2f}")
        st.info(f"Model: {selected_model_name} | Log-result: {log_prediction[0]:.4f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")