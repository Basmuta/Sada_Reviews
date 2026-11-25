from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "sample_reviews.csv"
MODEL_DIR = BASE / "models" / "sada_camelbert_finetuned"
MODEL_ID = os.getenv("SADA_MODEL_ID", "Basmuta/sada-camelbert-reviews")
MAX_UPLOAD_ROWS = 5000
CATEGORY_NAMES = {
    "login_issue": "Login & account access",
    "crash": "App crashes",
    "performance": "Performance",
    "payment": "Payments & billing",
    "usability": "Ease of use",
    "feature_request": "Feature requests",
    "positive_experience": "Positive experience",
}
PALETTE = ["#2563EB", "#0F766E", "#F59E0B", "#7C3AED", "#DB2777", "#0891B2", "#65A30D"]


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


@st.cache_resource(show_spinner="Loading the fine-tuned CAMeLBERT model…")
def load_camelbert():
    # Keep the project's Hugging Face model and prefer a local checkpoint when available.
    model_source = str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def label_name(label):
    return CATEGORY_NAMES.get(str(label), str(label).replace("_", " ").title())


def classify_reviews(reviews, tokenizer, model, device, batch_size=16):
    predictions, confidences, all_scores = [], [], []
    label_ids = list(range(model.config.num_labels))
    id2label = model.config.id2label
    labels = [id2label.get(i, id2label.get(str(i), str(i))) for i in label_ids]
    for start in range(0, len(reviews), batch_size):
        batch = [normalize_text(x) for x in reviews[start:start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        scores, ids = probs.max(dim=-1)
        predictions.extend(id2label.get(i, id2label.get(str(i), str(i))) for i in ids.cpu().tolist())
        confidences.extend(scores.cpu().tolist())
        all_scores.extend(probs.cpu().tolist())
    return predictions, confidences, all_scores, labels


def style_chart(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=58, b=12),
        font=dict(family="Arial, sans-serif", size=13, color="#25324B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        title_font=dict(size=17, color="#16233B"),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E9EEF5", zeroline=False, title=None)
    fig.update_yaxes(showgrid=False, title=None)
    return fig


st.set_page_config(page_title="Sada Reviews | App Review Insights", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    div[data-testid="stMetric"] {background:#F7F9FC;border:1px solid #E7ECF3;padding:16px 18px;border-radius:14px;}
    div[data-testid="stMetricLabel"] {color:#52627A;}
    .hero {padding: 22px 26px; border-radius: 18px; background: linear-gradient(110deg,#EFF6FF,#F7F5FF); border:1px solid #E3EAF7; margin-bottom: 18px;}
    .hero h1 {color:#182B49 !important;}
    .hero p {color:#53627A; margin-bottom:0; font-size:1.03rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>💬 Sada Reviews</h1><p>Turn app reviews into clear themes across Arabic dialects, English, and Arabizi.</p></div>',
    unsafe_allow_html=True,
)
st.caption("Fine-tuned CAMeLBERT · Review classification · Interactive insights")
sample = load_data()

try:
    tokenizer, model, device = load_camelbert()
    model_ready = True
    st.success("CAMeLBERT is ready. Enter one review or upload a CSV to analyze a batch.")
except Exception as exc:
    tokenizer = model = device = None
    model_ready = False
    st.error("The classifier could not load. Check that the Hugging Face model is public and includes its model and tokenizer files.")
    with st.expander("Technical details"):
        st.code(str(exc))

with st.sidebar:
    st.header("Analyze reviews")
    st.write("Classify one review or upload a CSV for batch analysis.")
    with st.form("single_review_form", clear_on_submit=False):
        user_review = st.text_area("Review text", placeholder="التطبيق ما يخليني اسجل دخول", height=130, max_chars=5000)
        analyze = st.form_submit_button("Classify review", type="primary", use_container_width=True)
    st.download_button(
        "Download CSV template",
        data="review\n\"The app keeps crashing when I sign in\"\n\"التطبيق يعلق كل ما فتحت الصفحة\"\n".encode("utf-8-sig"),
        file_name="sada_reviews_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.divider()
    uploaded_file = st.file_uploader("Upload reviews (CSV)", type=["csv"], help="Include a review or text column. Up to 5,000 rows per upload.")
    st.caption("Privacy: text submitted here is processed by this hosted app. Do not include names, phone numbers, account details, or confidential content.")

if analyze:
    if not user_review.strip():
        st.sidebar.warning("Enter a review first.")
    elif not model_ready:
        st.sidebar.warning("The model is unavailable right now, so classification cannot run.")
    else:
        with st.spinner("Analyzing your review…"):
            labels, confidences, score_rows, class_labels = classify_reviews([user_review], tokenizer, model, device)
        st.subheader("Your review result")
        result_cols = st.columns([1.25, 1, 1.8])
        result_cols[0].metric("Predicted theme", label_name(labels[0]))
        result_cols[1].metric("Model confidence", f"{confidences[0]:.1%}")
        result_cols[2].write("**Review**")
        result_cols[2].write(user_review)
        top_indices = sorted(range(len(class_labels)), key=lambda i: score_rows[0][i], reverse=True)[:3]
        top = pd.DataFrame({"Theme": [label_name(class_labels[i]) for i in top_indices], "Model score": [score_rows[0][i] for i in top_indices]})
        fig = px.bar(top.sort_values("Model score"), x="Model score", y="Theme", orientation="h", text=top.sort_values("Model score")["Model score"].map(lambda x: f"{x:.1%}"), color="Model score", color_continuous_scale="Blues", title="Top theme scores")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.12])
        style_chart(fig, 260)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if confidences[0] < 0.50:
            st.info("The model is uncertain about this review. Check the text or compare the top theme scores before using the result.")

st.divider()
st.subheader("Review insights")
st.write("Explore the bundled examples or analyze your own CSV. Use the filters to focus on a theme, language, or dialect.")

analysis = sample.copy()
if "review" not in analysis.columns and "text" in analysis.columns:
    analysis = analysis.rename(columns={"text": "review"})
analysis["category_name"] = analysis["category"].map(label_name)
analysis["source"] = "Sample dataset"

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        uploaded = pd.read_csv(BytesIO(file_bytes))
        review_col = "review" if "review" in uploaded.columns else ("text" if "text" in uploaded.columns else None)
        if review_col is None:
            st.error("This CSV needs a column named **review** or **text**. Download the template from the sidebar to get started.")
        elif len(uploaded) == 0:
            st.warning("The uploaded CSV has no rows.")
        elif len(uploaded) > MAX_UPLOAD_ROWS:
            st.error(f"This upload contains {len(uploaded):,} rows. Please upload {MAX_UPLOAD_ROWS:,} or fewer reviews at a time.")
        elif not model_ready:
            st.info("The model is unavailable, so uploaded reviews cannot be classified yet.")
        else:
            uploaded = uploaded.rename(columns={review_col: "review"}).copy()
            uploaded["review"] = uploaded["review"].fillna("").astype(str)
            uploaded = uploaded[uploaded["review"].str.strip().ne("")].reset_index(drop=True)
            digest = sha256(file_bytes).hexdigest()
            cache_key = f"batch_{digest}"
            if cache_key not in st.session_state:
                with st.spinner(f"Classifying {len(uploaded):,} reviews…"):
                    cats, confs, _, _ = classify_reviews(uploaded["review"].tolist(), tokenizer, model, device)
                uploaded["category"] = cats
                uploaded["confidence"] = confs
                st.session_state[cache_key] = uploaded
            analysis = st.session_state[cache_key].copy()
            analysis["category_name"] = analysis["category"].map(label_name)
            analysis["source"] = "Uploaded CSV"
            if len(analysis) == 0:
                st.warning("No non-empty reviews were found in the uploaded file.")
    except Exception as exc:
        st.error(f"Could not read this CSV. Check that it is a valid UTF-8 or standard CSV file. Details: {exc}")

if "category_name" not in analysis.columns:
    analysis["category_name"] = analysis["category"].map(label_name)

filter_col = "dialect" if "dialect" in analysis.columns else ("language" if "language" in analysis.columns else None)
filter1, filter2, filter3 = st.columns([1.3, 1, 1])
themes = sorted(analysis["category_name"].dropna().unique().tolist())
selected_themes = filter1.multiselect("Themes", themes, default=themes)
slice_values = sorted(analysis[filter_col].dropna().astype(str).unique().tolist()) if filter_col else []
selected_slices = filter2.multiselect(filter_col.title() if filter_col else "Group", slice_values, default=slice_values, disabled=not bool(filter_col))
min_conf = filter3.slider("Minimum confidence", 0, 100, 0, 5, format="%d%%", disabled="confidence" not in analysis.columns)

filtered = analysis[analysis["category_name"].isin(selected_themes)].copy()
if filter_col and selected_slices:
    filtered = filtered[filtered[filter_col].astype(str).isin(selected_slices)]
if "confidence" in filtered.columns:
    filtered = filtered[filtered["confidence"].fillna(0) >= min_conf / 100]

total = len(filtered)
unique_themes = filtered["category_name"].nunique()
avg_conf = filtered["confidence"].mean() if "confidence" in filtered.columns and total else None
top_theme = filtered["category_name"].value_counts().index[0] if total else "—"
kpis = st.columns(4)
kpis[0].metric("Reviews analyzed", f"{total:,}")
kpis[1].metric("Themes found", f"{unique_themes:,}")
kpis[2].metric("Most common theme", top_theme)
kpis[3].metric("Average confidence", f"{avg_conf:.1%}" if avg_conf is not None else "Sample labels")

if uploaded_file is None:
    st.info("You are viewing the project's illustrative sample dataset. It contains generated examples and should not be treated as real-world product feedback.")
elif "confidence" in analysis.columns:
    st.caption("Predictions are model estimates. Review low-confidence results before making product decisions.")

if total == 0:
    st.warning("No reviews match the current filters. Select more themes or groups.")
else:
    chart_left, chart_right = st.columns([1, 1.25], gap="large")
    counts = filtered["category_name"].value_counts().rename_axis("Theme").reset_index(name="Reviews")
    with chart_left:
        fig = px.bar(counts.sort_values("Reviews"), x="Reviews", y="Theme", orientation="h", text="Reviews", color="Reviews", color_continuous_scale=[[0, "#BFDBFE"], [1, "#1D4ED8"]], title="Review volume by theme")
        fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "array", "categoryarray": counts.sort_values("Reviews")["Theme"].tolist()}, xaxis_range=[0, max(counts["Reviews"].max() * 1.18, 1)])
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with chart_right:
        if filter_col:
            grouped = filtered.groupby([filter_col, "category_name"], observed=True).size().reset_index(name="Reviews")
            fig = px.bar(grouped, x=filter_col, y="Reviews", color="category_name", barmode="stack", title=f"Theme mix across {filter_col.lower()} groups", color_discrete_sequence=PALETTE, category_orders={filter_col: slice_values})
            fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=11)), xaxis_tickangle=-15)
            style_chart(fig, 430)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            share = counts.assign(Share=counts["Reviews"] / counts["Reviews"].sum())
            fig = px.pie(share, values="Reviews", names="Theme", hole=0.58, title="Share of reviews by theme", color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>%{value} reviews (%{percent})<extra></extra>")
            style_chart(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if "confidence" in filtered.columns and total:
        conf_left, conf_right = st.columns([1, 1.25], gap="large")
        with conf_left:
            fig = px.histogram(filtered, x="confidence", nbins=12, title="Prediction confidence", color_discrete_sequence=["#0F766E"])
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_yaxes(title="Reviews")
            style_chart(fig, 310)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with conf_right:
            st.markdown("#### What the confidence score means")
            st.write("The score is the model's estimated probability for its top theme. It is useful for prioritizing a human review, but it is not a guarantee that the prediction is correct.")

st.divider()
st.subheader("Explore classified reviews")
columns = [c for c in ["review", "category_name", "confidence", "language", "dialect", "sentiment"] if c in filtered.columns]
if columns and total:
    display_df = filtered[columns].rename(columns={"review": "Review", "category_name": "Predicted theme", "confidence": "Confidence", "language": "Language", "dialect": "Dialect", "sentiment": "Sentiment"})
    if "Confidence" in display_df:
        display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=380)
    st.download_button("Download filtered results", data=filtered.to_csv(index=False).encode("utf-8-sig"), file_name="sada_review_analysis.csv", mime="text/csv", type="primary")
else:
    st.caption("Upload a CSV to see predicted labels and confidence scores for individual reviews.")

with st.expander("About this project"):
    st.write("Sada Reviews uses a fine-tuned CAMeLBERT sequence classifier to organize app feedback into seven themes. It is designed as a lightweight analysis aid for Arabic dialects, English, and Arabizi. Check the predictions against real review context before acting on them.")
    st.caption("Model: Basmuta/sada-camelbert-reviews · Sample data is synthetic.")

