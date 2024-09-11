import streamlit as st

st.title("🎈 Diabetes APP")
st.write(
    "Test Drive Diabetes App"
)

st.slider("Pregnancies",0,17,(0,17))
st.slider("Glucose",0,199,(0,199))
st.slider("BloodPressure",0,122,(0,122))
st.slider("SkinThickness",0,99,(0,99))
st.slider("Insulin",0,846,(0,846))
st.slider("BMI",0.0,67.1,(0.0,67.1))
st.slider("DiabetesPedigreeFunction",0.078,2.42,(0.078,2.42))
st.slider("Age",21,81,(0,81))
