import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
model = joblib.load('model.joblib')

# Load the TF-IDF Vectorizer (if it was saved separately, otherwise create it)
tfidf = TfidfVectorizer(max_features=1000)  # Adjust max_features to your training setup

# Streamlit app
st.title("Spam Detection Classifier")
st.write("This app classifies whether a text message is spam or not.")

# Text input from the user
user_input = st.text_area("Enter a message for spam detection:")

# Preprocess input (using TF-IDF vectorizer as used during model training)
if user_input:
    # Transform input text using the TF-IDF vectorizer
    user_input_transformed = tfidf.transform([user_input])  # Must match how training data was processed
    
    # Predict using the loaded model
    prediction = model.predict(user_input_transformed)

    # Display the result
    if prediction == 1:
        st.write("This message is **SPAM**.")
    else:
        st.write("This message is **NOT SPAM**.")
