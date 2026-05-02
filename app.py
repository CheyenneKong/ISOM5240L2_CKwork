import streamlit as st
from transformers import pipeline
from PIL import Image
import os
from huggingface_hub import InferenceClient

# 1. Setup HF Inference Client for Gender (Remote API)
# Ensure you have 'HF_TOKEN' set in your environment variables
client = InferenceClient(
    api_key=os.environ.get("HF_TOKEN")
)

# 2. Setup Local Pipeline for Age (Local Model)
@st.cache_resource
def load_age_model():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_age_model()

st.title("🧑‍ identity Classifier")
st.write("Predicting Age (Local ViT) and Gender (HF API)")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    if st.button("Run Classification"):
        col1, col2 = st.columns(2)

        # --- Task 1: Gender Classification (API) ---
        with col1:
            st.subheader("Gender (via API)")
            try:
                # We seek the start of the file again to ensure the API reads it correctly
                uploaded_file.seek(0)
                gender_output = client.image_classification(
                    uploaded_file.read(), 
                    model="rizvandwiki/gender-classification-2"
                )
                top_gender = gender_output[0]
                st.success(f"Result: {top_gender['label']}")
                st.caption(f"Confidence: {top_gender['score']:.2%}")
            except Exception as e:
                st.error(f"API Error: {e}")

        # --- Task 2: Age Classification (Local) ---
        with col2:
            st.subheader("Age (via Local ViT)")
            with st.spinner("Calculating..."):
                age_results = age_classifier(image)
                top_age = age_results[0]
                st.info(f"Range: {top_age['label']}")
                st.caption(f"Confidence: {top_age['score']:.2%}")
