import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os

# Specify the model path
model_path = 'ai_detect_testv4.h5'

# Verify the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Please ensure it is in the correct location.")
    st.stop()
else:
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
    st.write("Model input shape:", model.input_shape)

# Preprocessing function
def preprocess_image(image):
    """Preprocess the uploaded image to match the model's input requirements."""
    image = image.resize((128, 128))  # Resize to (128, 128) as per the model's expected input
    image_array = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 128, 128, 3)
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

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Debugging: Check the processed image shape before predicting
    st.write("Processed image shape:", processed_image.shape)

    # Predict
    try:
        prediction = model.predict(processed_image)
        st.write("Model prediction raw output:", prediction)  # Debugging confidence scores
        
        # Interpret the result
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_idx] * 100
        label = "AI-Generated" if class_idx == 1 else "Not AI-Generated"
        
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
