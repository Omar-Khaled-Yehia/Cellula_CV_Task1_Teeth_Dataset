import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
# Load the saved model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

def predict_image(image, model):
    # Preprocess the image
    image = image.resize((192, 192))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class, predictions

# Streamlit app title
st.title("Teeth Classification Model")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image of a tooth", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when the button is clicked
    if st.button("Predict"):
        predicted_class, predictions = predict_image(image, model)
        st.write(f"Predicted Class: {class_names[predicted_class[0]]}")

