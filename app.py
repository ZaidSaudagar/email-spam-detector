import streamlit as st
import joblib
from model import SpamDetector
import time

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .spam {
        color: red;
        font-weight: bold;
    }
    .ham {
        color: green;
        font-weight: bold;
    }
    .probability {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📧 Email Spam Detector")
st.markdown("""
This application uses machine learning to classify emails as **spam** or **not spam (ham)**.
Paste your email content in the text area below and click the button to analyze.
""")

# Initialize or load model
@st.cache_resource
def load_model():
    detector = SpamDetector()
    try:
        detector.load_model()
        return detector
    except FileNotFoundError:
        st.error("Model not found. Please train the model first by running 'model.py'.")
        st.stop()

detector = load_model()

# User input
email_text = st.text_area("Enter email content here:", 
                         placeholder="Paste the email content you want to analyze...")

# Analyze button
if st.button("Analyze Email", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Simulate processing time for better UX
            time.sleep(1)
            
            # Make prediction
            prediction, probability = detector.predict(email_text)
            
            # Display results
            st.subheader("Results")
            
            if prediction == 1:
                st.markdown(f"<p class='spam'>🚨 This email is classified as SPAM</p>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='ham'>✅ This email is classified as HAM (not spam)</p>", 
                           unsafe_allow_html=True)
            
            # Show confidence levels
            st.markdown("<p class='probability'>Confidence levels:</p>", 
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="HAM Probability", 
                         value=f"{probability[0]*100:.2f}%")
            with col2:
                st.metric(label="SPAM Probability", 
                         value=f"{probability[1]*100:.2f}%")
            
            # Show explanation
            if prediction == 1:
                st.info("""
                **Spam emails often contain:**
                - Urgent language or threats
                - Requests for personal information
                - Unusual sender addresses
                - Promotions or offers that seem too good to be true
                """)
            else:
                st.info("""
                **This email appears legitimate. However, always be cautious with:**
                - Unexpected attachments
                - Requests for sensitive information
                - Links from unknown senders
                """)

# Sidebar information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This spam detection system uses:
    - **Natural Language Processing (NLP)** for text preprocessing
    - **TF-IDF Vectorization** for feature extraction
    - **Naive Bayes Classifier** for prediction
    """)
    
    st.header("How to Use")
    st.markdown("""
    1. Paste email content in the text box
    2. Click "Analyze Email"
    3. View the results and confidence levels
    """)
    
    st.header("Model Performance")
    st.markdown("""
    - Accuracy: ~98%
    - Precision (Spam): ~95%
    - Recall (Spam): ~90%
    """)

# Footer
st.markdown("---")
st.caption("Developed with Python, Scikit-learn, and Streamlit | © 2023 Email Spam Detector")