"""Optional Streamlit front end for the ECG heartbeat classification assignment."""
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model import make_model, softmax_probs  # noqa: E402

RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "ecgnet_best.pt"
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown / paced beat",
}
DESCRIPTIONS = {
    "N": "A heartbeat classified as normal.",
    "S": "An early beat originating above the ventricles.",
    "V": "A beat originating in the ventricles.",
    "F": "A beat combining normal and ventricular features.",
    "Q": "An unknown or paced heartbeat category.",
}

st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="♥", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1200px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 16px; background: linear-gradient(115deg,#10243a,#1c4960); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; font-size:2.15rem; margin:0 0 .35rem 0;}
.hero p {color:#d9e7ef; margin:0; font-size:1rem;}
.metric-card {padding: 1rem 1.1rem; border:1px solid rgba(128,128,128,.25); border-radius:13px;}
.small-muted {color: #8a96a3; font-size:.88rem;}
.result {padding:1rem 1.2rem; border-radius:12px; border:1px solid rgba(30,150,100,.4); background:rgba(30,150,100,.07);}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_results():
    with open(RESULTS / "test_results.json", encoding="utf-8") as f:
        test = json.load(f)
    with open(RESULTS / "data_info.json", encoding="utf-8") as f:
        data = json.load(f)
    with open(RESULTS / "best_model_info.json", encoding="utf-8") as f:
        best = json.load(f)
    return test, data, best

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = make_model(0)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def show_image(filename, caption):
    path = RESULTS / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} is not present in the results folder.")

def predict_signal(signal):
    model = load_model()
    if model is None:
        raise FileNotFoundError("The trained model file results/ecgnet_best.pt is missing.")
    x = torch.tensor(np.asarray(signal, dtype=np.float32).reshape(1, 187))
    with torch.no_grad():
        logits = model(x)
        probs = softmax_probs(logits)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs

test, data, best = load_results()
with st.sidebar:
    st.title("ECGNet")
    st.caption("Applied Machine Learning · Assignment 3")
    page = st.radio("Navigation", ["Overview", "Model performance", "Try a heartbeat", "About the project"])
    st.divider()
    st.caption("Research / educational demonstration")
    st.caption("Not a medical diagnostic tool.")

st.markdown("""
<div class="hero">
  <h1>ECG Heartbeat Classification</h1>
  <p>Explore the ECGNet model, review its evaluation results, and classify a 187-sample heartbeat signal.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Project overview")
    st.write("This project uses a fully connected neural network implemented manually in PyTorch to classify individual ECG heartbeat signals into five categories.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Test accuracy", f"{test['test_accuracy']*100:.2f}%")
    c2.metric("Balanced accuracy", f"{test['balanced_accuracy']*100:.2f}%")
    c3.metric("Trainable parameters", "32,645")
    c4.metric("Heartbeat classes", "5")
    st.divider()
    left,right = st.columns([1.15, .85])
    with left:
        st.subheader("Example ECG signals")
        show_image("example_beats.png", "One example heartbeat waveform from each class")
    with right:
        st.subheader("Classification categories")
        for label in CLASSES:
            st.markdown(f"**{label} — {CLASS_NAMES[label]}**")
            st.caption(DESCRIPTIONS[label])
    st.info("The dataset contains 187 signal values per heartbeat, sampled at 125 Hz. The model returns one of five class labels.")

elif page == "Model performance":
    st.subheader("Evaluation dashboard")
    c1,c2,c3 = st.columns(3)
    c1.metric("Test accuracy", f"{test['test_accuracy']*100:.2f}%")
    c2.metric("Balanced accuracy", f"{test['balanced_accuracy']*100:.2f}%")
    c3.metric("Always-normal baseline", f"{test['always_N_accuracy']*100:.2f}%")
    st.caption("Balanced accuracy averages the recall of all five classes, so minority classes are represented more fairly.")
    st.divider()
    st.subheader("Per-class accuracy")
    per = test["per_class_accuracy"]
    df = pd.DataFrame({"Heartbeat class": [f"{x} — {CLASS_NAMES[x]}" for x in CLASSES],
                       "Accuracy (%)": [per[x]*100 for x in CLASSES]})
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Heartbeat class"))
    st.subheader("Confusion matrix")
    cm = np.asarray(test["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    st.caption("Rows show the true class; columns show the model prediction.")
    st.dataframe(cm_df, use_container_width=True)
    show_image("confusion_matrix.png", "Confusion matrix for the held-out test set")
    with st.expander("Training curves and learning-rate selection"):
        show_image("lr_val_loss.png", "Validation loss for the three tested learning rates")
        a,b = st.columns(2)
        with a: show_image("loss_curves.png", "Training and validation loss")
        with b: show_image("accuracy_curves.png", "Training and validation accuracy")
        st.write(f"Selected learning rate: **{best['alpha']}** · Best epoch: **{best['best_epoch']}** · Validation loss: **{best['val_loss']:.4f}**")

elif page == "Try a heartbeat":
    st.subheader("Classify an ECG heartbeat")
    st.write("Upload a CSV containing exactly 187 numeric signal values. The file may contain one row or multiple rows; if multiple rows are supplied, the first row is classified.")
    st.warning("This is an educational machine-learning demonstration, not a medical device. Predictions must not be used to diagnose or make healthcare decisions.")
    uploaded = st.file_uploader("Upload heartbeat CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded, header=None)
            # Permit a single row of 187 values, or a standard dataset row with label in col 188.
            if frame.shape[1] == 188:
                st.caption("Detected 188 columns; treating the last column as a dataset label and using the first 187 signal values.")
                frame = frame.iloc[:, :187]
            if frame.shape[1] != 187:
                st.error(f"Expected 187 signal columns; received {frame.shape[1]}.")
            else:
                numeric = frame.apply(pd.to_numeric, errors="coerce")
                if numeric.isna().any().any():
                    st.error("The CSV contains missing or non-numeric values. Please upload a clean numeric signal.")
                else:
                    signal = numeric.iloc[0].to_numpy(dtype=np.float32)
                    if not np.isfinite(signal).all():
                        st.error("The signal contains non-finite values.")
                    else:
                        label, probs = predict_signal(signal)
                        st.markdown(f'<div class="result"><strong>Predicted class: {label} — {CLASS_NAMES[label]}</strong><br>{DESCRIPTIONS[label]}<br><span class="small-muted">Model confidence: {probs.max()*100:.2f}%</span></div>', unsafe_allow_html=True)
                        st.subheader("Class probabilities")
                        prob_df = pd.DataFrame({"Class": CLASSES, "Probability (%)": probs*100}).set_index("Class")
                        st.bar_chart(prob_df)
                        st.subheader("Uploaded ECG waveform")
                        chart_df = pd.DataFrame({"Amplitude": signal}, index=np.arange(1,188))
                        st.line_chart(chart_df)
                        if len(frame) > 1:
                            st.caption("Only the first row was classified. Upload individual files to inspect other rows.")
        except Exception as exc:
            st.error(f"Could not process this file: {exc}")
    else:
        st.info("No file uploaded yet. The input must contain 187 values representing one heartbeat.")

elif page == "About the project":
    st.subheader("How the system works")
    st.markdown("""
    1. **Data preparation:** 187 ECG samples are used as input features. The class label is stored separately.
    2. **Neural network:** 187 inputs → 128 ReLU units → 64 ReLU units → 5 output logits.
    3. **Loss:** softmax cross-entropy is implemented using `torch.logsumexp`.
    4. **Training:** mini-batch gradient descent with momentum and L2 weight decay; the learning rate is selected using validation loss.
    5. **Evaluation:** the held-out test set is evaluated once using accuracy, per-class accuracy, balanced accuracy, and a confusion matrix.
    """)
    st.subheader("Dataset")
    c1,c2,c3 = st.columns(3)
    c1.metric("Training rows", f"{data['train_rows']:,}")
    c2.metric("Validation rows", f"{data['val_rows']:,}")
    c3.metric("Test rows", f"{data['test_rows']:,}")
    st.write("Source dataset: ECG Heartbeat Categorization Dataset (MIT-BIH Arrhythmia-derived heartbeat segments). Each beat is represented by 187 samples at 125 Hz.")
    st.subheader("Network architecture")
    st.code("Input:  187 features\nHidden: 128 neurons (ReLU)\nHidden:  64 neurons (ReLU)\nOutput:   5 logits\nTrainable parameters: 32,645", language="text")
    st.caption("The front end presents the existing assignment outputs. It does not retrain the model or replace the original assignment scripts.")
