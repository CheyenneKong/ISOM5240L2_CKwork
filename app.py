import streamlit as st
from transformers import pipeline
from PIL import Image
import os
from huggingface_hub import InferenceClient

# 1. Setup HF Inference Client with your token
# Note: In production, use st.secrets or env vars for security!
HF_TOKEN = "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"

client = InferenceClient(
    api_key=HF_TOKEN
)

# 2. Setup Local Pipeline for Age (Caches the model to save time)
@st.cache_resource
def load_age_model():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_age_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Age & Gender Classifier", layout="centered")
st.title("🧑‍ identity Analysis")
st.write("Using local ViT for Age and HF API for Gender.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Prepare image for local model
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the image
    st.image(image, caption="Target Image", use_container_width=True)

    if st.button("Analyze Image"):
        col1, col2 = st.columns(2)

        # --- Task 1: Gender Classification (Remote API) ---
        with col1:
            st.subheader("Gender Prediction")
            try:
                # Reset file pointer to beginning for API read
                uploaded_file.seek(0)
                gender_output = client.image_classification(
                    uploaded_file.read(), 
                    model="rizvandwiki/gender-classification-2"
                )
                
                if gender_output:
                    top_gender = gender_output[0]
                    st.metric("Gender", top_gender['label'])
                    st.caption(f"Confidence: {top_gender['score']:.2%}")
            except Exception as e:
                st.error(f"API Error: {e}")

        # --- Task 2: Age Classification (Local Model) ---
        with col2:
            st.subheader("Age Prediction")
            with st.spinner("Analyzing Age..."):
                age_results = age_classifier(image)
                top_age = age_results[0]
                st.metric("Age Range", top_age['label'])
                st.caption(f"Confidence: {top_age['score']:.2%}")
                
        # Optional: Show full data
        with st.expander("Show Raw JSON Data"):
            st.write({"Gender_API_Response": gender_output, "Age_Local_Response": age_results})
