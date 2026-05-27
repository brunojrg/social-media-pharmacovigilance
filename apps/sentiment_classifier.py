import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model folder
MODEL_PATH = "../model"

# Load model + tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# UI
st.title("💊 Drug Review Sentiment Classifier")
st.write("Paste a drug review below and get its sentiment prediction.")

user_input = st.text_area("Enter review text:", height=150)

def predict(text):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = np.argmax(probs)

    return id2label[pred_class], probs

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        label, probs = predict(user_input)
        max_confidence = np.max(probs)


        st.subheader("Prediction:")
        
        if label == "Negative":
            bg_color = "#e1767f"   # light red
            text_color = "#440f15"
        elif label == "Neutral":
            bg_color = "#fff3cd"   # light yellow
            text_color = "#856404"
        else:  # Positive
            bg_color = "#aadbb5"   # light green
            text_color = "#155724"

        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                padding: 16px;
                border-radius: 10px;
                color: {text_color};
                font-size: 18px;
                font-weight: 500;
            ">
                Sentiment: {label}<br>
                Classification Confidence: {max_confidence:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )