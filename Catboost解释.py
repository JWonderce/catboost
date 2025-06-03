import streamlit as st  # Streamlit for creating the web application
import pandas as pd  # Pandas for data manipulation
import pickle  # Pickle for loading the trained model
import os  # OS for file path handling
import shap  # SHAP for model interpretability

# Load the model
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the model file
model_path = os.path.join(current_dir, 'catboost_model.pkl')
# Load the model from the file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Set the title of the Streamlit app
st.title("Prediction Model for Glycemic Control in Type 2 Diabetes")

# Sidebar input features using sliders with appropriate ranges and default values
diabetes_duration = st.sidebar.slider("Duration of Diabetes (1 = <5 years, 2 = 5–10 years, 3 = >10 years)", min_value=1, max_value=3, value=1, step=1)
cvd = st.sidebar.slider("Cardiovascular Disease (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0, step=1)
comorbidities = st.sidebar.slider("Number of Chronic Comorbidities", min_value=0, max_value=5, value=1, step=1)
neuropathy = st.sidebar.slider("Peripheral Neuropathy (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0, step=1)
sbp = st.sidebar.slider("Systolic Blood Pressure (SBP, mmHg)", min_value=80, max_value=200, value=120, step=5)
bmi = st.sidebar.slider("Body Mass Index (BMI, kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
ldl = st.sidebar.slider("Low-Density Lipoprotein (LDL-C, mmol/L)", min_value=1.0, max_value=5.2, value=2.6, step=0.1)
fpg = st.sidebar.slider("Fasting Plasma Glucose (FPG, mmol/L)", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
diet_score = st.sidebar.slider("Diet Compliance Score", min_value=0, max_value=10, value=5, step=1)
exercise_score = st.sidebar.slider("Exercise Compliance Score", min_value=0, max_value=10, value=5, step=1)
medication_score = st.sidebar.slider("Medication Adherence Score", min_value=0, max_value=10, value=5, step=1)
blood_sugar_monitoring_score = st.sidebar.slider("Blood Glucose Monitoring Score", min_value=0, max_value=10, value=5, step=1)
monthly_blood_sugar_checks = st.sidebar.slider("Monthly Frequency of Blood Glucose Testing", min_value=0, max_value=30, value=5, step=1)

# Create a DataFrame from the input features
input_data = pd.DataFrame({
    '糖尿病病程': [diabetes_duration],
    '心血管病变': [cvd],
    '慢性合并症数量': [comorbidities],
    '糖尿病周围神经病变': [neuropathy],
    'SBP': [sbp],
    'BMI': [bmi],
    'LDL-C': [ldl],
    'FPG': [fpg],
    '饮食标准分': [diet_score],
    '运动标准分': [exercise_score],
    '服药标准分': [medication_score],
    '血糖监测标准分': [blood_sugar_monitoring_score],
    '每月血糖检测次数': [monthly_blood_sugar_checks]
})

# Prediction button
if st.button("Predict"):
    # Make prediction using the model
    prediction = model.predict(input_data)
    st.write(f"Probability of Glycemic Control: {prediction[0]}")

    # Compute SHAP values for interpretability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    # Select SHAP values for the first instance
    sample_shap_values = shap_values[0]
    expected_value = explainer.expected_value

    # Construct SHAP Explanation object
    explanation = shap.Explanation(
        values=sample_shap_values.values,
        base_values=expected_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns.tolist()
    )

    # Save SHAP force plot as HTML
    shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))

    # Display the SHAP force plot in Streamlit
    st.subheader("SHAP Force Plot for Model Prediction")
    with open("shap_force_plot.html") as f:
        st.components.v1.html(f.read(), height=600)
