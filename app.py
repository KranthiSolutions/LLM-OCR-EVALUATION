import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="LLM and OCR Evaluation",
    layout="wide"
)

st.title("LLM and OCR Model Evaluation")

st.write(
    "Compare model extraction accuracy "
    "using verified sample documents."
)

uploaded_files = st.file_uploader(
    "Upload sample documents",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

selected_models = st.multiselect(
    "Select models to evaluate",
    [
        "Gemma 3",
        "Llama 4 Scout",
        "Nemotron OCR v2",
        "PaddleOCR-VL-1.6",
        "Gemini Flash",
        "Gemini Flash-Lite"
    ]
)

if st.button("Run Evaluation"):
    if not uploaded_files:
        st.warning(
            "Please upload at least one document."
        )

    elif not selected_models:
        st.warning(
            "Please select at least one model."
        )

    else:
        st.info(
            "Upload and model selection validated. "
            "Connect the evaluation engine to run tests."
        )

st.subheader("Model Accuracy")

# Example data only. Replace with actual test results.
demo_results = pd.DataFrame({
    "Model": [
        "Gemma 3",
        "Llama 4 Scout",
        "Nemotron OCR v2",
        "PaddleOCR-VL-1.6",
        "Gemini Flash",
        "Gemini Flash-Lite"
    ],
    "Accuracy": [90, 88, 92, 91, 94, 89]
})

fig = px.bar(
    demo_results,
    x="Model",
    y="Accuracy",
    title="Illustrative Model Accuracy Comparison",
    range_y=[0, 100]
)

st.plotly_chart(fig, use_container_width=True)