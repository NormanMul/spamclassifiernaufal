import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and TF-IDF vectorizer outside the main function to avoid re-loading on every request
@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    return model

@st.cache_resource
def load_vectorizer():
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust to the vectorizer you trained with
    return vectorizer

model = load_model()
vectorizer = load_vectorizer()

# Streamlit app
st.title("Spam Detection Classifier")
st.write("This app classifies whether a text message is spam or not.")

# Text input from the user
user_input = st.text_area("Enter a message for spam detection:")

# Preprocess input using the cached TF-IDF vectorizer and model
if user_input:
    # Transform input text using the TF-IDF vectorizer
    user_input_transformed = vectorizer.transform([user_input])
    
    # Predict using the loaded model
    prediction = model.predict(user_input_transformed)

    # Display the result
    if prediction == 1:
        st.write("This message is **SPAM**.")
    else:
        st.write("This message is **NOT SPAM**.")
