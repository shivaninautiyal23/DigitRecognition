import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import os

# ----------------------------
# LOAD MODEL
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "mnist_cnn.h5")

if not os.path.exists(model_path):
    st.error(f"Model not found at: {model_path}")
    st.stop()

model = tf.keras.models.load_model(model_path)

# ----------------------------
# UI
# ----------------------------
st.title("🖊️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click Predict")

# ----------------------------
# DRAWING CANVAS
# ----------------------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",  # white digit
    background_color="#000000",  # black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype("uint8")

        # RGBA → grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # ✅ DO NOT invert (already correct format)

        # Normalize pixel values
        gray = gray.astype("float32") / 255.0

        # Find bounding box
        coords = cv2.findNonZero((gray > 0.2).astype("uint8"))

        if coords is None:
            st.warning("Draw something properly 😄")
            st.stop()

        x, y, w, h = cv2.boundingRect(coords)

        digit = gray[y:y+h, x:x+w]

        # Make square
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.float32)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

        # Add padding (important for MNIST look)
        square = cv2.copyMakeBorder(
            square, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0
        )

        # Resize to 28x28
        final_img = cv2.resize(square, (28, 28))

        # Reshape for model
        input_img = final_img.reshape(1, 28, 28, 1)

        # ----------------------------
        # DEBUG (optional)
        # ----------------------------
        st.image(final_img, caption="Processed Image", width=120)

        # Predict
        prediction = model.predict(input_img)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"✏️ Predicted Digit: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}")

    else:
        st.warning("Draw something first 😄")
