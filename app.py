import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and TF-IDF vectorizer outside the main function to avoid re-loading on every request
@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    return model

@st.cache_resource
def load_vectorizer():
    # Assuming the vectorizer was saved as 'vectorizer.joblib'
    vectorizer = joblib.load('vectorizer.joblib')  # Load the saved TF-IDF vectorizer
    return vectorizer

# Load the model and vectorizer only once
model = load_model()
vectorizer = load_vectorizer()

# Streamlit app UI
st.title("Spam Detection Classifier")
st.write("This app classifies whether a text message is spam or not.")

# User input for the text message
user_input = st.text_area("Enter a message for spam detection:")

# Preprocess input and make a prediction
if user_input:
    # Transform the input using the loaded TF-IDF vectorizer
    user_input_transformed = vectorizer.transform([user_input])
    
    # Predict with the loaded model
    prediction = model.predict(user_input_transformed)

    # Display the prediction result
    if prediction == 1:
        st.write("This message is **SPAM**.")
    else:
        st.write("This message is **NOT SPAM**.")
