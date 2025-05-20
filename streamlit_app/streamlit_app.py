import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ========== CONFIG ==========
MODEL_PATH = "iris_model.pkl"

# ========== FUNCIÓN PARA CARGAR EL MODELO ==========
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # Entrenar solo si no existe (por primera vez)
        iris = load_iris()
        X, y = iris.data, iris.target
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
    return model

# ========== CARGAR MODELO ==========
model = load_model()
iris = load_iris()
class_names = iris.target_names

# ========== INTERFAZ ==========
st.title("Clasificador de Flores 🌸 (Iris Dataset)")
st.markdown("Introduce las características de la flor para predecir su especie.")

# Inputs
sepal_length = st.slider("Longitud del sépalo (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Longitud del pétalo (cm)", 1.0, 7.0, 4.35)
petal_width = st.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.3)

# Predicción
if st.button("Predecir especie"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    predicted_class = class_names[prediction]

    st.success(f"🌼 Especie predicha: **{predicted_class.capitalize()}**")