import streamlit as st
from transformers import pipeline

# 1. Cache the model so it doesn't reload on every click
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def main():
    st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")
    
    # Load the cached model
    sentiment_pipeline = load_pipeline()

    st.title("🎭 Sentiment Analysis Tool")
    st.markdown("This app uses a **DistilBERT** model to determine if your text is positive or negative.")

    # 2. Better UI layout
    user_input = st.text_area("Enter text to analyze:", placeholder="I had a great day today!")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip() != "":
            with st.spinner("Analyzing..."):
                result = sentiment_pipeline(user_input)
                sentiment = result[0]["label"]
                confidence = result[0]["score"]

                # 3. Dynamic styling based on result
                if sentiment == "POSITIVE":
                    st.success(f"**Sentiment:** {sentiment} (Score: {confidence:.2%})")
                else:
                    st.error(f"**Sentiment:** {sentiment} (Score: {confidence:.2%})")
        else:
            st.warning("Please enter some text first.")

if __name__ == "__main__":
    main()
