import streamlit as st
import joblib

# Load the model and vectorizer
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')  # Load the trained model

@st.cache_resource
def load_vectorizer():
    return joblib.load('vectorizer.joblib')  # Load the fitted vectorizer

# Load model and vectorizer once
model = load_model()
vectorizer = load_vectorizer()

# Password Protection
st.sidebar.title("Login")
password = st.sidebar.text_input("Enter the password:", type="password")

# Check the password
if password == '#Ne3wSp4m!':
    st.sidebar.success("Access Granted!")
    
    # Streamlit app UI for spam detection
    st.title("Spam Detection Classifier")
    st.write("This app classifies whether a text message is spam or not. (Made by Naufal Prawiro")

    # User input for the text message
    user_input = st.text_area("Enter a message for spam detection:")

    # Submit button
    if st.button("Submit"):
        if user_input:
            # Preprocess input and make a prediction
            user_input_transformed = vectorizer.transform([user_input])
            
            # Predict with the loaded model
            prediction = model.predict(user_input_transformed)

            # Display the prediction result
            if prediction == 1:
                st.write("This message is **SPAM**.")
            else:
                st.write("This message is **NOT SPAM**.")
        else:
            st.write("Please enter a message.")
else:
    st.sidebar.error("Incorrect password. Access Denied!")
