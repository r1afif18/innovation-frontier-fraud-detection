# ==============================================================================
# STREAMLIT APPLICATION FOR LINGUISTIC FRAUD DETECTION
# ==============================================================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# --- PAGE CONFIGURATION ---
# Sets the basic configuration for the web page, like the title in the browser tab and the icon.
st.set_page_config(
    page_title="Linguistic Fraud Detector",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FUNCTIONS & MODEL LOADING ---

# Uses Streamlit's cache to prevent reloading the model on every interaction.
# This makes the app much faster.
@st.cache_resource
def load_assets():
    """Loads the trained model and vectorizer from the saved_models folder."""
    try:
        model_path = os.path.join('saved_models', 'model.pkl')
        vectorizer_path = os.path.join('saved_models', 'tfidf_vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Ensures NLTK stopwords are available in the deployment environment.
        try:
            stopwords.words('indonesian')
        except LookupError:
            nltk.download('stopwords')

        return model, vectorizer
    except FileNotFoundError:
        # Displays a clear error message if the model files are not found.
        st.error("Error: Model or vectorizer files not found. Please ensure 'model.pkl' and 'tfidf_vectorizer.pkl' are in the 'saved_models/' folder.")
        return None, None

# Text preprocessing function.
# IMPORTANT: This function must be 100% identical to the one used during model training.
def preprocess_text(text, stopwords_list):
    """Cleans and processes the user's input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords_list and len(word) > 2)
    return text.strip()

# Load assets (model & vectorizer) when the app first starts.
model, tfidf_vectorizer = load_assets()
if model and tfidf_vectorizer:
    # Prepare the list of stopwords for use by the preprocess_text function.
    stopwords_indonesia = stopwords.words('indonesian')
    stopwords_indonesia.extend(['yg', 'dg', 'dgn', 'dr', 'kpd', 'utk', 'kak', 'gan', 'sis'])

# --- USER INTERFACE (UI) STREAMLIT ---

# Title and Description
st.title("🔎 Linguistic Fraud Fingerprint Detector")
st.markdown("""
This application uses a machine learning model to analyze the linguistic style of a message and predict whether it is potentially fraudulent. Simply enter the text below and click the "Analyze Message" button.
""")

st.warning("**Disclaimer:** This model is a predictive tool based on text patterns and is **not legal proof**. Always verify information independently before taking any action.")

# User Text Input Area
user_input = st.text_area("Enter the suspicious text message to analyze:", height=150, placeholder="Example: Congratulations! Your number has won a 100jt prize, please contact the admin...")

# Button to Trigger Analysis
if st.button("Analyze Message", type="primary"):
    if user_input and model and tfidf_vectorizer:
        # Process only if there is input and the model loaded successfully
        
        # 1. Apply preprocessing to the user's input
        clean_input = preprocess_text(user_input, stopwords_indonesia)
        
        # 2. Transform the clean text into a numerical vector using TF-IDF
        input_vector = tfidf_vectorizer.transform([clean_input])
        
        # 3. Make a prediction and get the probabilities
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)
        
        # Display the main prediction result
        st.subheader("Analysis Result:")
        
        col1, col2 = st.columns([0.6, 0.4]) # Adjust column widths
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **FRAUD DETECTED**")
            else:
                st.success("✅ **NOT DETECTED AS FRAUD**")
        
        with col2:
            # Display the risk score as a clear metric
            risk_score = prediction_proba[0][1] * 100
            st.metric(label="Fraud Risk Score", value=f"{risk_score:.2f}%")

        # Provide a progress bar visualization for the risk score
        st.progress(int(risk_score) / 100)
        
        # Provide a contextual explanation based on the risk score
        if risk_score > 75:
            st.write("Interpretation: The model is highly confident that this message is fraudulent. Please be very careful and do not provide personal information or make any transfers.")
        elif risk_score > 40:
            st.write("Interpretation: The model has detected some features similar to known fraudulent patterns. It's best to be cautious and not click on any links.")
        else:
            st.write("Interpretation: The model did not find strong linguistic patterns that indicate fraud in this message.")

    elif not user_input:
        st.warning("Please enter text to analyze.")
    # If the model fails to load, the error message will be displayed by the load_assets function

# Adding additional information at the bottom using tabs
st.write("---")
st.subheader("Additional Information")

tabs = st.tabs(["How It Works", "Model Performance", "About the Project"])

with tabs[0]:
    st.markdown("""
    This application works in a few steps:
    1.  **Text Preprocessing:** The text you enter is cleaned of punctuation, numbers, and common words (stopwords).
    2.  **Feature Extraction (TF-IDF):** The clean text is converted into a numerical representation that the model can understand. The model measures the importance of each word in your text compared to thousands of other examples it has learned from.
    3.  **Classification (Logistic Regression):** This numerical vector is then fed into a classification model trained on over a thousand examples of fraudulent and legitimate messages.
    4.  **Risk Score:** The model outputs a probability, which we convert into a 0-100 score to indicate the model's confidence in its prediction.
    """)

with tabs[1]:
    st.markdown("""
    This model was evaluated using metrics relevant to fraud detection cases. Here is a sample of the model's performance on a test dataset:
    - **F1-Score (Fraud Class):** 0.92 (Excellent at balancing finding fraud cases and not making false accusations).
    - **Recall (Fraud Class):** 0.95 (Meaning the model successfully identified 95% of all actual fraud cases in the test data).
    - **Precision (Fraud Class):** 0.89 (Meaning that of everything predicted as fraud, 89% of it was actually fraud).
    
    *Note: You can replace these metrics with the results from your notebook and even add your confusion matrix image.*
    """)
    # Example for adding an image: st.image('path/to/your/confusion_matrix.png')

with tabs[2]:
    st.markdown("""
    This project is part of the **"Innovation Frontiers"** case study, which aims to apply data science to complex and "anti-mainstream" social problems in Indonesia.
    
    - **Created by:** Rafif Sudanta
    - **View Code:** [GitHub Repository](https://github.com/r1afif18/innovation-frontier-fraud-detection)
    """)
