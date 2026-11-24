from pathlib import Path
import re
import os
import torch
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "sample_reviews.csv"
MODEL_DIR = BASE / "models" / "sada_camelbert_finetuned"
MODEL_ID = os.getenv("SADA_MODEL_ID", "Basmuta/sada-camelbert-reviews")
CATEGORY_NAMES = {
    "login_issue": "Login and account access", "crash": "Crashes",
    "performance": "Performance", "payment": "Payments and billing",
    "usability": "Usability", "feature_request": "Feature request",
    "positive_experience": "Positive experience",
}

def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " [EMAIL] ", text)
    text = re.sub(r"(?<!\d)(?:05\d{8}|\+9665\d{8})(?!\d)", " [PHONE] ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_camelbert():
    # Prefer local files; otherwise download the model from the Hugging Face Hub.
    model_source = str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def classify_reviews(reviews, tokenizer, model, device, batch_size=16):
    labels, confidences = [], []
    for start in range(0, len(reviews), batch_size):
        batch = [normalize_text(x) for x in reviews[start:start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        scores, ids = probs.max(dim=-1)
        for idx, score in zip(ids.cpu().tolist(), scores.cpu().tolist()):
            labels.append(model.config.id2label.get(idx, model.config.id2label.get(str(idx), str(idx))))
            confidences.append(score)
    return labels, confidences

st.set_page_config(page_title="Sada Reviews", page_icon="📊", layout="wide")
st.title("Sada Reviews")
st.caption("Fine-tuned CAMeLBERT for Arabic app review classification across dialects, English, and Arabizi.")
sample = load_data()

try:
    tokenizer, model, device = load_camelbert()
    st.success("Primary classifier: CAMeLBERT-Mix fine-tuned on project review categories.")
except Exception as exc:
    tokenizer = model = device = None
    st.warning("Could not load the fine-tuned model. Confirm that the Hugging Face repository is public and contains the model and tokenizer files.")
    with st.expander("Model loading details"):
        st.code(str(exc))

with st.sidebar:
    st.header("Analyze a review")
    user_review = st.text_area("Review text", placeholder="التطبيق ما يخليني اسجل دخول")
    analyze = st.button("Classify review", type="primary", use_container_width=True)
    st.divider()
    uploaded_file = st.file_uploader("Or upload a CSV with a review column", type=["csv"])
    st.caption("Inference runs locally; review text is not sent to an external service.")

if analyze and user_review.strip():
    if model is None:
        st.info("Fine-tune CAMeLBERT in Colab first to enable classification.")
    else:
        labels, scores = classify_reviews([user_review], tokenizer, model, device)
        st.subheader("CAMeLBERT prediction")
        c1, c2 = st.columns(2)
        c1.metric("Review theme", CATEGORY_NAMES.get(labels[0], labels[0]))
        c2.metric("Model confidence", f"{scores[0]:.0%}")

analysis = sample.copy()
if uploaded_file is not None:
    try:
        uploaded = pd.read_csv(uploaded_file)
        if "review" not in uploaded.columns:
            st.error("The CSV must include a review column.")
        elif model is None:
            st.info("Add the fine-tuned checkpoint to classify uploaded reviews.")
        else:
            analysis = uploaded.copy()
            analysis["review"] = analysis["review"].fillna("").astype(str)
            analysis["category"], analysis["confidence"] = classify_reviews(analysis["review"].tolist(), tokenizer, model, device)
            analysis["category_name"] = analysis["category"].map(CATEGORY_NAMES).fillna(analysis["category"])
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")

st.divider()
st.subheader("Review theme dashboard")
if uploaded_file is None:
    st.info("The bundled sample is synthetic and does not represent real user reviews.")
if "category_name" not in analysis.columns:
    analysis["category_name"] = analysis["category"].map(CATEGORY_NAMES).fillna(analysis["category"])
left, right = st.columns(2)
counts = analysis["category_name"].value_counts().rename_axis("Review theme").reset_index(name="Review count")
with left:
    fig = px.bar(counts, x="Review count", y="Review theme", orientation="h", color="Review count", color_continuous_scale="Blues", title="Reviews by theme")
    fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
with right:
    if "dialect" in analysis.columns:
        slice_col = "dialect"
    elif "language" in analysis.columns:
        slice_col = "language"
    else:
        slice_col = None
    if slice_col:
        by_slice = analysis.groupby([slice_col, "category_name"]).size().reset_index(name="Review count")
        fig = px.bar(by_slice, x=slice_col, y="Review count", color="category_name", title="Themes by dialect / language", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.metric("Review count", len(analysis))

st.caption(f"Records shown: {len(analysis):,}")
with st.expander("View classified data"):
    columns = [c for c in ["review", "category_name", "confidence", "language", "dialect"] if c in analysis.columns]
    st.dataframe(analysis[columns], use_container_width=True, hide_index=True)

