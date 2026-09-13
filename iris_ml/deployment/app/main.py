import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Iris Lab", page_icon="🌿", layout="centered")
st.caption("PMLDL · ASSIGNMENT 1 · DEPLOYMENT")
st.title("🌿 Iris Lab")
st.write("Определите вид ириса по четырём измерениям цветка.")
st.caption("Все значения — в сантиметрах. Пример ниже соответствует Iris setosa.")

with st.form("prediction"):
    left, right = st.columns(2)
    with left:
        sepal_length = st.number_input("Длина чашелистика", 0.1, 30.0, 5.1, 0.1)
        petal_length = st.number_input("Длина лепестка", 0.1, 30.0, 1.4, 0.1)
    with right:
        sepal_width = st.number_input("Ширина чашелистика", 0.1, 30.0, 3.5, 0.1)
        petal_width = st.number_input("Ширина лепестка", 0.1, 30.0, 0.2, 0.1)
    submitted = st.form_submit_button("Определить вид", type="primary", use_container_width=True)

if submitted:
    try:
        with st.spinner("Модель обрабатывает измерения…"):
            response = requests.post(f"{API_URL}/predict", json={
                "sepal_length": sepal_length, "sepal_width": sepal_width,
                "petal_length": petal_length, "petal_width": petal_width,
            }, timeout=15)
            response.raise_for_status()
            result = response.json()
        st.success(f"Вид: Iris {result['species']}")
        st.bar_chart({"Вид": list(result["probabilities"]),
                      "Вероятность": list(result["probabilities"].values())},
                     x="Вид", y="Вероятность", horizontal=True)
        st.caption(f"Версия модели (MLflow run): {result['run_id']}")
    except (requests.RequestException, ValueError, KeyError):
        st.error("API временно недоступен. Дождитесь завершения развёртывания и повторите запрос.")

st.divider()
st.caption("DVC → StandardScaler + LogisticRegression → FastAPI → Streamlit")
