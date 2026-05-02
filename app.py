import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# Set up the app title and layout
st.set_page_config(page_title="Age Classifier", page_icon="🎂")
st.title("🎂 Age Classification using ViT")
st.write("Upload an image to predict the age range of the person.")

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_classifier():
    # Loading the Vision Transformer (ViT) age classifier
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

# Initialize the classifier
age_classifier = load_classifier()

# File uploader for user images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for a cleaner UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Classifying..."):
            # Classify age
            age_predictions = age_classifier(image)
            
            # Sort predictions by score (highest first)
            age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
            
            # Display top result
            top_prediction = age_predictions[0]
            st.metric(label="Predicted Age Range", value=top_prediction['label'])
            st.write(f"**Confidence:** {top_prediction['score']:.2%}")
            
    # Show all probabilities in a chart below
    with st.expander("See detailed probabilities"):
        # Convert to DataFrame for better chart rendering
        df = pd.DataFrame(age_predictions)
        st.bar_chart(data=df.set_index('label'))
