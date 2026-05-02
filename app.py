import streamlit as st
from transformers import pipeline
from PIL import Image

# Set up the app title and layout
st.title("🎂 Age Classification using ViT")
st.write("Upload an image to predict the age range of the person.")

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_classifier()

# File uploader for user images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Classifying..."):
        # Classify age
        age_predictions = age_classifier(image)
        
        # Sort predictions by score (highest first)
        age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
        
        # Display results
        top_prediction = age_predictions[0]
        st.success(f"**Predicted Age Range: {top_prediction['label']}**")
        st.write(f"Confidence Score: {top_prediction['score']:.2%}")
        
        # Optional: Show all probabilities in a chart
        with st.expander("See detailed probabilities"):
            labels = [p['label'] for p in age_predictions]
            scores = [p['score'] for p in age_predictions]
            st.bar_chart(data=dict(zip(labels, scores)))
