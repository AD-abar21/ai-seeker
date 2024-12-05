import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Debugging: Display current directory and available files
st.write("Current directory:", os.getcwd())
st.write("Available files:", os.listdir())

# Specify the model path
loaded_model = tf.keras.layers.TFSMLayer('ai_detect_testv3.keras', call_endpoint='serving_default')  # Update this to the actual path if necessary

# Verify the model file exists
if not os.path.exists(loaded_model):
    st.error(f"Model file not found: {loaded_model}. Please ensure it is in the correct location.")
else:
    # Load the Keras model
    model = tf.keras.models.load_model(loaded_model)
    st.success("Model loaded successfully!")

    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((224, 224))  # Adjust size based on your model
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    # Streamlit interface
    st.title("AI Image Detection")
    st.write("Upload an image to predict if it is AI-generated or not.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        # Interpret the result
        is_ai_generated = prediction[0][0] > 0.5  # Change this based on your model's output
        label = "AI-Generated" if is_ai_generated else "Not AI-Generated"
        st.write(f"Prediction: {label}")
