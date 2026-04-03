from pathlib import Path
import streamlit as st

from explainability import find_suspicious_claims, generate_explanation
from model import get_top_keywords, get_training_metrics, load_metrics, predict_news, save_metrics, train_model
from source_manager import is_source_trusted

METRICS_PATH = Path("metrics.json")


def initialize_metrics() -> None:
    if not METRICS_PATH.exists():
        save_metrics({"articles_analyzed": 0, "training": {}})


def render_metric_card(label: str, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_json_card(title: str, data):
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    st.json(data)


initialize_metrics()

st.set_page_config(page_title="Fake News Detection", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071120 0%, #0b1f3a 45%, #123766 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2433 0%, #171d2b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 3.2rem;
        padding-bottom: 2.5rem;
        padding-left: 3.2rem;
        padding-right: 3.2rem;
    }

    .main-title {
        font-size: 52px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }

    .sub-title {
        font-size: 18px;
        color: #d1d9e6;
        margin-bottom: 2.4rem;
        line-height: 1.6;
        max-width: 900px;
    }

    .section-title {
        font-size: 36px;
        font-weight: 750;
        color: #ffffff;
        margin-top: 0.8rem;
        margin-bottom: 1.4rem;
        line-height: 1.2;
    }

    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 14px;
        margin-top: 4px;
    }

    .result-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 10px 32px rgba(0,0,0,0.20);
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 22px 16px;
        text-align: center;
        box-shadow: 0 8px 26px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #cdd8ea;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 34px;
        font-weight: 800;
    }

    .pill-real, .pill-fake, .pill-true, .pill-false {
        display: inline-block;
        padding: 10px 18px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 12px;
    }

    .pill-real, .pill-true {
        background: rgba(34,197,94,0.18);
        color: #bbf7d0;
        border: 1px solid rgba(34,197,94,0.35);
    }

    .pill-fake {
        background: rgba(239,68,68,0.18);
        color: #fecaca;
        border: 1px solid rgba(239,68,68,0.35);
    }

    .pill-false {
        background: rgba(245,158,11,0.18);
        color: #fde68a;
        border: 1px solid rgba(245,158,11,0.35);
    }

    .confidence-text {
        font-size: 18px;
        color: #e2e8f0;
        margin-bottom: 12px;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.10);
        margin-top: 32px;
        margin-bottom: 32px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        color: #ffffff !important;
        font-weight: 600;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #cbd5e1 !important;
    }

    div[data-testid="stButton"] button {
        width: 220px;
        border-radius: 12px;
        height: 48px;
        font-weight: 700;
        color: white;
        border: none;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 18px rgba(37,99,235,0.35);
    }

    div[data-testid="stButton"] button:hover {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }

    .sidebar-metric {
        color: #dbeafe;
        font-size: 15px;
        margin-bottom: 10px;
    }

    .sidebar-value {
        color: #86efac;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Sidebar
with st.sidebar:
    st.markdown("## Model Controls")
    if st.button("Retrain Model"):
        with st.spinner("Training model..."):
            train_model()
        st.success("Model retrained successfully.")

    training_metrics = get_training_metrics()
    st.markdown("---")
    st.markdown(
        f'<div class="sidebar-metric">Latest Training Accuracy: <span class="sidebar-value">{training_metrics.get("accuracy", "N/A") if training_metrics else "N/A"}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sidebar-metric">Latest Training F1: <span class="sidebar-value">{training_metrics.get("f1", "N/A") if training_metrics else "N/A"}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sidebar-metric">Transformer Enabled: <span class="sidebar-value">{training_metrics.get("transformer_enabled", False) if training_metrics else False}</span></div>',
        unsafe_allow_html=True,
    )

# Header
st.markdown('<div class="main-title">Fake News Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hybrid model using TF-IDF word features, character features, transformer embeddings, source trust, and explainability.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Analyze News Article</div>', unsafe_allow_html=True)

title = st.text_input(
    "News Title",
    placeholder="Example: ISRO launches new Earth observation satellite"
)

text = st.text_area(
    "News Article",
    height=220,
    placeholder="Paste the full article or headline here..."
)

source = st.text_input(
    "News Source",
    placeholder="Example: Reuters, BBC, The Hindu"
)

analyze = st.button("Analyze Article")

with st.expander("Preview Output Structure"):
    st.json(
        {
            "prediction": {"prediction": "real/fake", "confidence": 0.0, "probabilities": {"real": 0.0, "fake": 0.0}},
            "suspicious_claims": ["sample phrase"],
            "explanation": ["sample explanation"],
            "source_trusted": True,
            "top_terms": ["term1", "term2"],
            "system_metrics": {"articles_analyzed": 0},
        }
    )

# Analysis results
if analyze:
    if not title.strip() and not text.strip():
        st.warning("Enter at least a title or article text.")
    else:
        with st.spinner("Analyzing article..."):
            prediction_data = predict_news(title=title, text=text, source=source)
            suspicious = find_suspicious_claims(f"{title} {text}")
            keywords = [word for word, _ in get_top_keywords(title=title, text=text, source=source, top_n=8)]
            trusted = is_source_trusted(source)

            explanation = generate_explanation(
                title=title,
                text=text,
                source=source,
                prediction=prediction_data["prediction"],
                confidence=prediction_data["confidence"],
                important_terms=keywords,
            )

            metrics = load_metrics()
            metrics["articles_analyzed"] = int(metrics.get("articles_analyzed", 0)) + 1
            save_metrics(metrics)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

        # Prediction
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)
        pred_class = "pill-fake" if prediction_data["prediction"] == "fake" else "pill-real"
        pred_text = prediction_data["prediction"].upper()
        st.markdown(f'<span class="{pred_class}">{pred_text}</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="confidence-text">Confidence Score: <b>{prediction_data["confidence"]:.4f}</b></div>',
            unsafe_allow_html=True,
        )
        st.json(prediction_data)
        st.markdown('</div>', unsafe_allow_html=True)

        # Explanation
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_json_card("Explanation", explanation)
        st.markdown('</div>', unsafe_allow_html=True)

        # Suspicious Claims
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_json_card("Suspicious Claims", suspicious if suspicious else ["No suspicious claims found"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Source Trust
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Source Trust Verification</div>', unsafe_allow_html=True)
        source_class = "pill-true" if trusted else "pill-false"
        source_text = "TRUSTED" if trusted else "NOT TRUSTED"
        st.markdown(f'<span class="{source_class}">{source_text}</span>', unsafe_allow_html=True)
        st.write(f"Entered Source: **{source if source.strip() else 'Not Provided'}**")
        st.markdown('</div>', unsafe_allow_html=True)

        # Top Terms
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_json_card("Top Important Terms", keywords if keywords else ["No important terms found"])
        st.markdown('</div>', unsafe_allow_html=True)

        # System Metrics
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_json_card("System Metrics", load_metrics())
        st.markdown('</div>', unsafe_allow_html=True)

# Footer metrics
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Training Performance</div>', unsafe_allow_html=True)

training_metrics = get_training_metrics()
if training_metrics:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Accuracy", training_metrics.get("accuracy", "N/A"))
    with c2:
        render_metric_card("Precision", training_metrics.get("precision", "N/A"))
    with c3:
        render_metric_card("Recall", training_metrics.get("recall", "N/A"))
    with c4:
        render_metric_card("F1 Score", training_metrics.get("f1", "N/A"))
else:
    st.info("Training metrics are not available yet.")