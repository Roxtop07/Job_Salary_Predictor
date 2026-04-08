import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. SET UP PAGE ---
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction App")

# --- SALARY ADJUSTMENT LOGIC ---
# Define high-tech roles that typically pay more
HIGH_TECH_ROLES = [
    'Machine Learning Engineer', 'AI Engineer', 'Data Scientist', 
    'Cloud Engineer', 'DevOps Engineer', 'Cybersecurity Analyst'
]

# Define high-paying locations
HIGH_PAY_LOCATIONS = ['USA', 'Singapore', 'Australia', 'UK', 'Germany']
LOW_PAY_LOCATIONS = ['India', 'Remote']

# --- CURRENCY AND COUNTRY-SPECIFIC SALARY DATA ---
COUNTRY_CONFIG = {
    'USA': {
        'currency': 'USD', 'symbol': '$', 'rate': 1.0,
        'salary_factor': 1.0,  # Base reference
        'avg_tech_salary': 120000
    },
    'India': {
        'currency': 'INR', 'symbol': '₹', 'rate': 83.5,
        'salary_factor': 0.25,  # India salaries ~25% of US in USD terms
        'avg_tech_salary': 1500000  # 15 LPA
    },
    'UK': {
        'currency': 'GBP', 'symbol': '£', 'rate': 0.79,
        'salary_factor': 0.85,
        'avg_tech_salary': 65000
    },
    'Germany': {
        'currency': 'EUR', 'symbol': '€', 'rate': 0.92,
        'salary_factor': 0.80,
        'avg_tech_salary': 70000
    },
    'Canada': {
        'currency': 'CAD', 'symbol': 'C$', 'rate': 1.36,
        'salary_factor': 0.75,
        'avg_tech_salary': 95000
    },
    'Australia': {
        'currency': 'AUD', 'symbol': 'A$', 'rate': 1.53,
        'salary_factor': 0.90,
        'avg_tech_salary': 130000
    },
    'Singapore': {
        'currency': 'SGD', 'symbol': 'S$', 'rate': 1.34,
        'salary_factor': 0.85,
        'avg_tech_salary': 100000
    },
    'Netherlands': {
        'currency': 'EUR', 'symbol': '€', 'rate': 0.92,
        'salary_factor': 0.75,
        'avg_tech_salary': 65000
    },
    'Sweden': {
        'currency': 'SEK', 'symbol': 'kr', 'rate': 10.5,
        'salary_factor': 0.70,
        'avg_tech_salary': 650000
    },
    'Remote': {
        'currency': 'USD', 'symbol': '$', 'rate': 1.0,
        'salary_factor': 0.80,  # Remote typically pays slightly less
        'avg_tech_salary': 100000
    }
}

def convert_to_local_currency(usd_salary, location):
    """Convert USD salary to local currency based on location."""
    config = COUNTRY_CONFIG.get(location, COUNTRY_CONFIG['USA'])
    local_salary = usd_salary * config['rate']
    return local_salary, config['currency'], config['symbol']

def adjust_salary_for_country(base_salary, location):
    """Adjust salary based on country's market rates."""
    config = COUNTRY_CONFIG.get(location, COUNTRY_CONFIG['USA'])
    adjusted = base_salary * config['salary_factor']
    return adjusted

def calculate_experience_level(experience_years, skills_count, education_level, certifications):
    """Determine experience level based on multiple factors."""
    score = 0
    
    # Experience contribution (max 40 points)
    score += min(experience_years * 2, 40)
    
    # Skills contribution (max 20 points)
    score += min(skills_count, 20)
    
    # Education contribution (max 20 points)
    edu_scores = {'High School': 5, 'Diploma': 10, 'Bachelor': 12, 'Master': 16, 'PhD': 20}
    score += edu_scores.get(education_level, 10)
    
    # Certifications contribution (max 10 points)
    score += min(certifications * 2, 10)
    
    return score

def apply_salary_adjustments(base_salary, input_data):
    """
    Apply realistic salary adjustments based on candidate profile.
    Entry-level candidates with minimal qualifications get lower salaries.
    """
    adjusted_salary = base_salary
    
    # Calculate experience score
    exp_score = calculate_experience_level(
        input_data['experience_years'],
        input_data['skills_count'],
        input_data['education_level'],
        input_data['certifications']
    )
    
    # --- ENTRY-LEVEL PENALTY ---
    # If very entry level (0 years, few skills), apply reduction
    if input_data['experience_years'] == 0:
        adjusted_salary *= 0.75  # 25% reduction for no experience
        
    if input_data['skills_count'] <= 2:
        adjusted_salary *= 0.85  # 15% reduction for minimal skills
    
    # Entry-level score penalty (score < 25 is considered entry-level)
    if exp_score < 25:
        entry_penalty = 0.6 + (exp_score / 25) * 0.4  # Scale from 0.6 to 1.0
        adjusted_salary *= entry_penalty
    
    # --- ROLE-BASED ADJUSTMENTS ---
    # Non-tech roles typically pay less at entry level
    if input_data['job_title'] not in HIGH_TECH_ROLES:
        if input_data['experience_years'] <= 2:
            adjusted_salary *= 0.90  # 10% less for non-tech entry roles
    
    # --- COMPANY SIZE ADJUSTMENTS ---
    if input_data['company_size'] == 'Startup' and input_data['experience_years'] <= 2:
        adjusted_salary *= 0.90  # Startups pay less for entry level
    
    # --- MINIMUM SALARY FLOOR (in USD) ---
    # Ensure salary doesn't go below realistic minimum
    min_salary = 25000  # $25K USD minimum (will be converted later)
    adjusted_salary = max(adjusted_salary, min_salary)
    
    return adjusted_salary, exp_score

# --- 2. DYNAMIC PATH RESOLUTION ---
# This finds the folder where app.py is located, even on the Streamlit Cloud server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_single_model(model_name):
    """Loads only the selected model to save RAM."""
    file_map = {
        "Linear Regression": "linear_regression_pipeline.pkl",
        "Decision Tree": "decision_tree_pipeline.pkl",
        "Random Forest": "random_forest_pipeline.pkl"
    }
    
    # Construct the full absolute path
    file_path = os.path.join(BASE_DIR, file_map[model_name])
    
    if not os.path.exists(file_path):
        # Debugging: Show what files ARE there if it fails
        available_files = os.listdir(BASE_DIR)
        raise FileNotFoundError(
            f"Could not find '{file_map[model_name]}' in {BASE_DIR}. "
            f"Files present in folder: {available_files}"
        )
        
    return joblib.load(file_path)

# Sidebar for model selection
selected_model_name = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Linear Regression", "Decision Tree", "Random Forest"]
)

# Load ONLY the selected model
try:
    with st.spinner(f"Loading {selected_model_name}... (This may take a moment for large files)"):
        model = load_single_model(selected_model_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Tip: Ensure your .pkl files were pushed using Git LFS and are in the same folder as app.py.")
    st.stop()

# --- 3. UI INPUTS ---
st.subheader("Enter Job Details")

with st.expander("ℹ️ How does this predictor work?"):
    st.write("""
    This application predicts your estimated annual salary based on various professional metrics.
    
    **How to use:**
    1. **Select a Machine Learning Model** from the sidebar (Linear Regression, Decision Tree, or Random Forest).
    2. **Fill in your details** below, including your job title, experience, education, and industry.
    3. Click **Predict Salary** to get an estimate.
    
    *Note: The predictions are based on historical data patterns and should be used as a reference rather than a definitive salary expectation.*
    """)

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

    input_df = pd.DataFrame([input_dict])

    try:
        log_prediction = model.predict(input_df)
        
        # Convert log back to actual value (USD base)
        base_salary_usd = np.expm1(log_prediction[0])
        
        # Apply realistic adjustments for entry-level candidates
        adjusted_salary_usd, exp_score = apply_salary_adjustments(base_salary_usd, input_dict)
        
        # Apply country-specific salary adjustment
        country_adjusted_usd = adjust_salary_for_country(adjusted_salary_usd, location)
        
        # Convert to local currency
        local_salary, currency_code, currency_symbol = convert_to_local_currency(country_adjusted_usd, location)
        
        st.balloons()
        
        # Display salary in local currency
        if location == 'India':
            # Show in Lakhs format for India
            lakhs = local_salary / 100000
            if lakhs >= 1:
                st.success(f"### Predicted Annual Salary: ₹{lakhs:,.2f} Lakhs")
                st.caption(f"(₹{local_salary:,.0f} per year)")
            else:
                st.success(f"### Predicted Annual Salary: ₹{local_salary:,.0f}")
        else:
            st.success(f"### Predicted Annual Salary: {currency_symbol}{local_salary:,.2f} {currency_code}")
        
        # Show USD equivalent for non-USD countries
        if currency_code != 'USD':
            st.caption(f"💵 *Equivalent: ${country_adjusted_usd:,.2f} USD*")
        
        # Show experience level indicator
        if exp_score < 25:
            level = "🟢 Entry Level"
        elif exp_score < 50:
            level = "🟡 Mid Level"
        elif exp_score < 75:
            level = "🟠 Senior Level"
        else:
            level = "🔴 Expert Level"
        
        st.info(f"**Model:** {selected_model_name} | **Experience Level:** {level} | **Location:** {location}")
        
        # Show adjustment note for entry-level
        if exp_score < 25:
            st.caption("💡 *Salary adjusted for entry-level profile based on experience, skills, and qualifications.*")
        
        # Show country salary context
        country_config = COUNTRY_CONFIG.get(location, COUNTRY_CONFIG['USA'])
        avg_salary = country_config['avg_tech_salary']
        st.caption(f"📊 *Average tech salary in {location}: {currency_symbol}{avg_salary:,} {currency_code}*")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")