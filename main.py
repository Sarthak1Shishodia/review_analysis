import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}
 
model = load_model("simplernn_model.h5", compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

#streamlit
import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment:")

user_input = st.text_area("Review Text")
if st.button('classify'):
    preprocessed_input = preprocess_text(user_input)
    predeiction = model.predict(preprocessed_input)
    sentiment = 'Positive' if predeiction[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {predeiction[0][0]:.4f}")
else:
    st.write("Please enter a review .")

