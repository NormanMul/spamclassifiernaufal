import streamlit as st
import joblib

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')  # Load the trained model

@st.cache_resource
def load_vectorizer():
    return joblib.load('vectorizer.joblib')  # Load the fitted vectorizer

# Load model and vectorizer once
model = load_model()
vectorizer = load_vectorizer()

# Streamlit app UI
st.title("Spam Detection Classifier")
st.write("This app classifies whether a text message is spam or not.")

# User input for the text message
user_input = st.text_area("Enter a message for spam detection:")

# Preprocess input and make a prediction
if user_input:
    # Transform the input using the loaded, fitted TF-IDF vectorizer
    user_input_transformed = vectorizer.transform([user_input])
    
    # Predict with the loaded model
    prediction = model.predict(user_input_transformed)

    # Display the prediction result
    if prediction == 1:
        st.write("This message is **SPAM**.")
    else:
        st.write("This message is **NOT SPAM**.")
