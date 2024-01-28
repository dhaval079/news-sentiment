import streamlit as st
import pandas as pd
from Final import *

# Load the model and other necessary components
def main():
    st.title("Sentiment Prediction App")

    # Input text from user
    user_input = st.text_area("Enter a text for sentiment prediction")

    # Make prediction when the user clicks the button
    if st.button("Predict Sentiment"):
        prediction_result = make_prediction(user_input)
        st.success(f"Predicted Sentiment: {prediction_result}")

if __name__ == "__main__":
    main()
