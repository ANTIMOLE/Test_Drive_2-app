import streamlit as st
import pickle
import os

st.title("🎈 Diabetes APP")
st.write(
    "Test Drive Diabetes App"
)

pregnancy = st.slider("Pregnancies",0,17,(0,17))
Glucose = st.slider("Glucose",0,199,(0,199))
BloodPressure = st.slider("BloodPressure",0,122,(0,122))
SkinThickness = st.slider("SkinThickness",0,99,(0,99))
Insulin = st.slider("Insulin",0,846,(0,846))
BMI = st.slider("BMI",0.0,67.1,(0.0,67.1))
DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFunction",0.078,2.42,(0.078,2.42))
Age = st.slider("Age",21,81,(0,81))

import os
file_path = 'Model/knn_dt_diabetes_model.pkl'
st.write("File path exists:", os.path.exists(file_path))

file_path = 'Model/knn_dt_diabetes_model.pkl'
with open(file_path,'rb') as file:
    model = pickle.load(file)
st.write("load success")

#result = loaded_model.score(, y_test)