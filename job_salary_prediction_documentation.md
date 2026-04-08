# Job Salary Prediction - Complete Documentation

## Project Overview
This project builds machine learning models to predict job salaries based on various features like job title, experience, education, and more.

---

## Table of Contents
1. [Import Libraries](#1-import-required-libraries)
2. [Load Dataset](#2-load-the-dataset)
3. [Data Exploration](#3-data-exploration)
4. [Data Visualization](#4-data-visualization)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Model Training](#6-model-training)
7. [Model Export](#7-save-trained-models)

---

## 1. Import Required Libraries

```python
import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

**Purpose**: Import essential Python libraries for data analysis and visualization:
- **pandas**: Data manipulation and analysis
- **matplotlib.pyplot**: Data visualization
- **numpy**: Numerical computing
- **seaborn**: Statistical data visualization

---

## 2. Load the Dataset

```python
df = pd.read_csv('job_salary_prediction_dataset.csv')
```

**Purpose**: Load the job salary prediction dataset from a CSV file into a pandas DataFrame for further analysis.

---

## 3. Data Exploration

### 3.1 Dataset Information
```python
df.info()
```
**Purpose**: Display comprehensive information about the DataFrame including column names, data types, non-null counts, and memory usage.

### 3.2 Dataset Shape
```python
df.shape
```
**Purpose**: Get the dimensions of the dataset (number of rows and columns).

### 3.3 Statistical Summary
```python
df.describe()
```
**Purpose**: Generate descriptive statistics for numerical columns including count, mean, standard deviation, min, max, and quartile values.

### 3.4 Preview Data
```python
df.head()
```
**Purpose**: Display the first 5 rows of the dataset to understand its structure and content.

---

## 4. Data Visualization

### 4.1 Salary Distribution - Box Plot
```python
df.boxplot(column='salary')
plt.show()
```
**Purpose**: Visualize the distribution of salary values using a box plot to identify:
- Median salary
- Interquartile range (IQR)
- Potential outliers

### 4.2 Experience Years Distribution
```python
df.boxplot(column='experience_years')
plt.show()
```
**Purpose**: Visualize the distribution of experience years using a box plot to check for outliers.

### 4.3 Skills Count Distribution
```python
df.boxplot(column='skills_count')
plt.show()
```
**Purpose**: Visualize the distribution of skills count using a box plot to understand the spread of skills among employees.

### 4.4 Certifications Distribution
```python
df.boxplot(column='certifications')
plt.show()
```
**Purpose**: Visualize the distribution of certifications using a box plot to identify any anomalies.

---

## 5. Data Preprocessing

### 5.1 Outlier Treatment - Winsorization
```python
from scipy.stats.mstats import winsorize
df['salary'] = winsorize(df['salary'], limits=[0.01, 0.05])
```
**Purpose**: Apply winsorization to handle extreme outliers in the salary column:
- Lower 1% of values are capped
- Upper 5% of values are capped

This technique reduces the impact of outliers without removing data points.

### 5.2 Log Transformation of Salary
```python
df['salary'] = np.log1p(df['salary'])
```
**Purpose**: Apply log transformation (`log1p`) to the salary column to:
- Reduce skewness in the target variable
- Make the distribution more normal
- Improve model performance for linear algorithms

### 5.3 Missing Values Check
```python
df.isna().sum()
```
**Purpose**: Check for any missing (NaN) values in each column of the dataset.

### 5.4 Duplicate Records Check
```python
df.duplicated().sum()
```
**Purpose**: Count the number of duplicate rows in the dataset to ensure data quality.

### 5.5 Feature-Target Split
```python
from sklearn.model_selection import train_test_split
X = df.drop('salary', axis=1)
y = df['salary']
```
**Purpose**: Separate the dataset into:
- **X (Features)**: All columns except salary
- **y (Target)**: The salary column we want to predict

### 5.6 Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
```
**Purpose**: Split the data into training (80%) and testing (20%) sets:
- `random_state=42` ensures reproducibility
- Training set is used to build models
- Test set is used to evaluate model performance

### 5.7 Column Validation
```python
expected_cols = [
    'job_title', 'company_size', 'location', 
    'education_level', 'remote_work', 
    'experience_years', 'skills_count'
]
missing = [col for col in expected_cols if col not in X_train.columns]
```
**Purpose**: Verify that all expected columns are present in the training data before building the preprocessing pipeline.

### 5.8 Data Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('Ohe', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         ['job_title', 'company_size', 'location', 'industry']),
        ('Ordinal', OrdinalEncoder(), ['education_level', 'remote_work']),
        ('scaler', StandardScaler(), ['experience_years', 'skills_count'])
    ],
    remainder='passthrough'
)
```
**Purpose**: Create a comprehensive preprocessing pipeline using `ColumnTransformer`:
- **OneHotEncoder**: Convert categorical variables (job_title, company_size, location, industry) into binary features
- **OrdinalEncoder**: Encode ordinal variables (education_level, remote_work) preserving order
- **StandardScaler**: Scale numerical features (experience_years, skills_count) to have zero mean and unit variance

---

## 6. Model Training

### 6.1 Linear Regression
```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
```
**Purpose**: Build and train a Linear Regression model with the preprocessing pipeline.

### 6.2 Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor
pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
]) 
pipeline_dt.fit(X_train, y_train)
y_pred_dt = pipeline_dt.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
```
**Purpose**: Build and evaluate a Decision Tree model:
- Uses the same preprocessor for consistent feature transformation
- Decision trees can capture non-linear relationships
- Metrics reported: Mean Squared Error (MSE) and R² Score

### 6.3 Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
```
**Purpose**: Build and evaluate a Random Forest model:
- Ensemble method combining multiple decision trees
- Generally more robust and less prone to overfitting
- Metrics reported: Mean Squared Error (MSE) and R² Score

---

## 7. Save Trained Models

```python
import joblib

joblib.dump(pipeline, "linear_regression_pipeline.pkl")
joblib.dump(pipeline_dt, "decision_tree_pipeline.pkl")
joblib.dump(pipeline_rf, "random_forest_pipeline.pkl")
```

**Purpose**: Export all trained model pipelines as pickle files for:
- Future predictions without retraining
- Deployment in production environments
- Model versioning and reproducibility

---

## Model Files Generated
| File Name | Model Type |
|-----------|------------|
| `linear_regression_pipeline.pkl` | Linear Regression |
| `decision_tree_pipeline.pkl` | Decision Tree Regressor |
| `random_forest_pipeline.pkl` | Random Forest Regressor |

---

## Key Techniques Used

| Technique | Purpose |
|-----------|---------|
| **Winsorization** | Cap extreme outliers |
| **Log Transformation** | Normalize skewed target variable |
| **OneHotEncoding** | Convert nominal categories to binary |
| **OrdinalEncoding** | Preserve order in ordinal categories |
| **StandardScaler** | Normalize numerical features |
| **Pipeline** | Chain preprocessing and modeling steps |

---

## Author
Generated from `job_salary_prediction_dataset.ipynb`
