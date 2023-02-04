import tensorflow as tf
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

st.title(
    """Handwritten Digit recognition By Soham Patel"""
)

model = tf.keras.models.load_model(
    r'../handwritten_digit_recognization/tfmodel/')


canvas_data = st_canvas(
    background_color="#000000",
    stroke_color="#FFFFFF",
    stroke_width=15,
    update_streamlit=True,
    height=300,
    width=300
)


if canvas_data.image_data is not None:
    data1 = canvas_data.image_data
    data = data1.copy()
    data = data.astype('uint8')
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = cv2.resize(data, (28, 28))
    data = data / 255.0
    print(data)
    st.image(data)
    temp = (model.predict(data.reshape(1, 28, 28, 1)))
    st.title(np.argmax(temp))
