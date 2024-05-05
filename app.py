import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache
def load_model():
  model = tf.keras.models.load_model('vgg.hdf5')
  return model

st.write("""   # Coffee Bean Quality Prediction App""")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
import numpy as np
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.array(image, dtype = 'float32')/255.0
    #img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model=load_model())
    class_names = ['Dark', 'Green', 'Light', 'Medium']
    string = "This image most likely is: " + class_names[np.argmax(predictions)]
    st.success(string)
