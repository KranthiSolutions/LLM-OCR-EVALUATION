import json
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.document_loader import extract_pdf_text, pdf_to_images
from src.evaluation import compare_values, flatten_json
from src.model_client import run_extraction


 #App Configuration

st.set_page_config(
    page_title="Model Evaluation Framework(POC)",
    layout="wide"
)

st.title("Model Evaluation Framework (POC)")

st.caption(
    "Sample Model Testing Platform"

)

# 2. SECURE API KEY CONFIGURATION
# Load local .env file if available.
load_dotenv(override=False)


def get_secret(name):
    """
    Retrieve API credentials securely.

    Priority:
    1. Streamlit Secrets
    2. Environment variables / local .env
    """

    try:
        value = st.secrets.get(name)

        if value:
            return str(value)

    except (FileNotFoundError, KeyError):
        pass

    return os.getenv(name)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


# LiteLLM reads these environment variables.
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


# MODELS
DEFAULT_MODELS = [
    "gemini/gemini-3.8-flash",
    "openai/gpt-4.1-mini",
    "ollama/llama4:scout",
    "openai/gpt-4.1",
    "paddleocr-vl-1.6",
]


with st.sidebar:
    st.header("Evaluation settings")
    model_ids_text = st.text_area(
        "Model IDs (one per line)",
        value="\n".join(DEFAULT_MODELS),
        height=180,
        help=(
            "Enter the exact provider/model IDs supported "
            "by your configured integrations."
        )
    )

    model_ids = [
        x.strip()
        for x in model_ids_text.splitlines()
        if x.strip()
    ]

    selected_models = st.multiselect(
        "Models to run",
        options=model_ids,
        default=model_ids[:2]
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )

    max_tokens = st.number_input(
        "Maximum output tokens",
        min_value=256,
        max_value=8000,
        value=1800,
        step=256
    )

    st.divider()

    st.subheader("API Configuration")

    if GEMINI_API_KEY:
        st.success("Gemini API configured")
    else:
        st.warning("Gemini API key missing")

    if OPENAI_API_KEY:
        st.success("OpenAI API configured")
    else:
        st.warning("OpenAI API key missing")

    st.caption(
        "API keys are loaded server-side and are not displayed."
    )



# DOCUMENT UPLOAD
st.subheader("1. Upload documents and ground truth")

col1, col2 = st.columns(2)

with col1:

    uploaded_files = st.file_uploader(
        "Upload synthetic healthcare PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

with col2:

    gt_file = st.file_uploader(
        "Upload ground_truth.json (from the sample pack)",
        type=["json"]
    )


if not uploaded_files:

    st.info(
        "Upload one or more synthetic PDF documents to begin."
    )

    st.stop()


if not gt_file:

    st.warning(
        "Upload ground_truth.json so the application "
        "can calculate accuracy."
    )

    st.stop()


if not selected_models:

    st.warning(
        "Select at least one model in the sidebar."
    )

    st.stop()



# LOAD GROUND TRUTH

try:

    ground_truth = json.load(gt_file)

    gt_by_filename = {
        d["filename"]: d
        for d in ground_truth.get("documents", [])
    }

except Exception:

    st.error(
        "Could not read ground_truth.json. "
        "Verify that the file contains valid JSON "
        "and the expected document structure."
    )

    st.stop()


# DOCUMENT PREVIEW


st.subheader("2. Review selected test files")

preview_rows = []

for f in uploaded_files:

    matched = f.name in gt_by_filename

    preview_rows.append({
        "filename": f.name,
        "ground_truth_match": matched,
        "size_kb": round(f.size / 1024, 1)
    })


preview_df = pd.DataFrame(preview_rows)

st.dataframe(
    preview_df,
    use_container_width=True,
    hide_index=True
)


missing_ground_truth = [
    f.name
    for f in uploaded_files
    if f.name not in gt_by_filename
]

if missing_ground_truth:

    st.warning(
        f"{len(missing_ground_truth)} document(s) "
        "do not have matching ground-truth entries. "
        "These documents will be skipped."
    )



# RUN EVALUATION
if st.button("Run evaluation", type="primary"):

    all_results = []
    field_results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(uploaded_files) * len(selected_models)

    done = 0

    for uploaded in uploaded_files:

        expected_doc = gt_by_filename.get(uploaded.name)

        if not expected_doc:

            st.warning(
                f"No ground-truth entry for "
                f"{uploaded.name}; skipped."
            )

            done += len(selected_models)

            progress.progress(
                min(done / total, 1.0)
            )

            continue

        pdf_bytes = uploaded.getvalue()

        try:

            extracted_text = extract_pdf_text(
                pdf_bytes
            )

            # Use page images when no text is extracted.
            page_images = (
                pdf_to_images(pdf_bytes)
                if not extracted_text.strip()
                else []
            )

        except Exception:

            st.error(
                f"Could not process {uploaded.name}. "
                "Verify that it is a valid PDF."
            )

            done += len(selected_models)

            progress.progress(
                min(done / total, 1.0)
            )

            continue

        for model_id in selected_models:

            status.write(
                f"Running {model_id} "
                f"on {uploaded.name}..."
            )

            start = time.perf_counter()

            try:

                # ------------------------------------------
                # MODEL EXTRACTION
                # ------------------------------------------

                predicted = run_extraction(
                    model_id=model_id,
                    document_text=extracted_text,
                    document_type=expected_doc.get(
                        "document_type",
                        "healthcare_document"
                    ),
                    page_images=page_images,
                    temperature=temperature,
                    max_tokens=int(max_tokens),
                )

                elapsed = (
                    time.perf_counter() - start
                )

                # ------------------------------------------
                # GROUND TRUTH COMPARISON
                # ------------------------------------------

                expected_fields = expected_doc.get(
                    "fields",
                    {}
                )

                flat_expected = flatten_json(
                    expected_fields
                )

                flat_predicted = flatten_json(
                    predicted
                )

                comparisons = compare_values(
                    flat_expected,
                    flat_predicted
                )

                accuracy = (
                    sum(
                        1
                        for x in comparisons
                        if x["match"]
                    ) / len(comparisons)
                    if comparisons
                    else 0.0
                )

                # ------------------------------------------
                # SAVE MODEL RESULT
                # ------------------------------------------

                all_results.append({
                    "document": uploaded.name,
                    "model": model_id,
                    "status": "success",
                    "field_accuracy_pct": round(
                        accuracy * 100,
                        2
                    ),
                    "fields_compared": len(
                        comparisons
                    ),
                    "latency_seconds": round(
                        elapsed,
                        2
                    ),
                    "error": "",
                    "raw_json": json.dumps(
                        predicted,
                        ensure_ascii=False
                    ),
                })

                # ------------------------------------------
                # FIELD LEVEL RESULTS
                # ------------------------------------------

                for item in comparisons:

                    field_results.append({
                        "document": uploaded.name,
                        "model": model_id,
                        "field": item["field"],
                        "expected": item["expected"],
                        "predicted": item["predicted"],
                        "match": item["match"],
                    })

            except Exception:

                elapsed = (
                    time.perf_counter() - start
                )

                all_results.append({
                    "document": uploaded.name,
                    "model": model_id,
                    "status": "error",
                    "field_accuracy_pct": None,
                    "fields_compared": 0,
                    "latency_seconds": round(
                        elapsed,
                        2
                    ),
                    "error": (
                        "Model execution failed. "
                        "Check API credentials, "
                        "model availability, "
                        "and server logs."
                    ),
                    "raw_json": "",
                })

            done += 1

            progress.progress(
                min(done / total, 1.0)
            )

    status.write("Evaluation finished.")

    if all_results:

        results_df = pd.DataFrame(
            all_results
        )

        st.session_state[
            "results_df"
        ] = results_df

        st.session_state[
            "field_results_df"
        ] = pd.DataFrame(
            field_results,
            columns=[
                "document",
                "model",
                "field",
                "expected",
                "predicted",
                "match"
            ]
        )



# EVALUATION RESULTS

if "results_df" in st.session_state:

    st.subheader("3. Results")

    results_df = st.session_state[
        "results_df"
    ]

    success_df = results_df[
        results_df["status"] == "success"
    ]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Model/document runs",
        len(results_df)
    )

    m2.metric(
        "Successful runs",
        len(success_df)
    )

    average_accuracy = (
        f"{success_df['field_accuracy_pct'].mean():.1f}%"
        if not success_df.empty
        else "N/A"
    )

    m3.metric(
        "Average field accuracy",
        average_accuracy
    )

    st.dataframe(
        results_df.drop(
            columns=["raw_json"]
        ),
        use_container_width=True,
        hide_index=True
    )



    # MODEL COMPARISON


    if not success_df.empty:

        st.subheader(
            "4. Model Performance Comparison"
        )

        chart_df = (
            success_df.groupby(
                "model",
                as_index=False
            )
            .agg(
                average_accuracy=(
                    "field_accuracy_pct",
                    "mean"
                ),
                average_latency=(
                    "latency_seconds",
                    "mean"
                )
            )
        )

        st.bar_chart(
            chart_df.set_index(
                "model"
            )["average_accuracy"]
        )

        st.dataframe(
            chart_df,
            use_container_width=True,
            hide_index=True
        )



    # FIELD LEVEL COMPARISON
   

    with st.expander(
        "Field-by-field comparison"
    ):

        fields_df = st.session_state[
            "field_results_df"
        ]

        st.dataframe(
            fields_df,
            use_container_width=True,
            hide_index=True
        )

    # DOWNLOAD RESULTS


    st.download_button(
        "Download results CSV",
        data=results_df.drop(
            columns=["raw_json"]
        ).to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "llm_ocr_evaluation_results.csv"
        ),
        mime="text/csv"
    )

    st.download_button(
        "Download field comparison CSV",
        data=st.session_state[
            "field_results_df"
        ].to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "llm_ocr_field_comparison.csv"
        ),
        mime="text/csv"
    )



# FOOTER

st.caption(
    "Scores are exact-match scores after basic "
    "normalization. They are not clinical quality "
    "scores. Review all outputs before use."
)