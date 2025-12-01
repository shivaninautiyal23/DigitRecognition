'''use this if file not found error is shown'''
# cd C:/Users/Rajesh/OneDrive/Desktop/PYTHON/handwriting    
# streamlit run hndwrt.py
 
 
                                               
import streamlit as st 
import tensorflow as tf
from PIL import Image , ImageOps
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2


import os
st.write("Current working directory:",  os.getcwd())
st.write("Files in the folder:",  os.listdir())


# Load the trained CNN model
model_path = "C:/Users/Rajesh/OneDrive/Desktop/PYTHON/handwriting/mnist_cnn.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("not found")
    st.stop()

st.title("🖊️ Handwriting Digit Recognition")
st.write("Draw a digit (0-9) below and click **Predict**")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Resize image to 28x28 and convert to grayscale
        img = canvas_result.image_data

        img = cv2.resize(img, (28, 28))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray / 255.0  # Normalize

        # Reshape to match model input
        input_img = gray.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(input_img)
        predicted_label = np.argmax(prediction)

        st.success(f"✏️ Predicted Digit: **{predicted_label}**")
