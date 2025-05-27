import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load model
with open('Diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar: App summary
st.sidebar.title("Diabetes Prediction App")
st.sidebar.markdown("""
This app uses a **Random Forest Classifier** to predict the likelihood of a person having diabetes based on:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The model was trained on the **PIMA Indian Diabetes Dataset**.
""")

# App title
st.title("Diabetes Prediction")

# Input form
st.header("Enter Patient Data")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 33)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Low risk of diabetes (Probability: {prediction_proba:.2f})")

# Feature importance
st.subheader("Feature Importance")

feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
importances = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(feature_names, importances, color='skyblue')
ax.set_xlabel("Importance")
ax.set_title("Feature Importance from Random Forest")
st.pyplot(fig)
