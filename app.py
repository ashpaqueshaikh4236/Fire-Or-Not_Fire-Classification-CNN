import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title('Fire or not_Fire Classifier')
model = tf.keras.models.load_model('model.keras')



def preprocess_image(image):
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_image = st.file_uploader('Select an image', type=['jpg', 'jpeg', 'png'])



if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.resize((256, 256))
    st.image(image)

    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)
    st.write(prediction)

    if prediction[0][0] > 0.5:
        st.success('Prediciton: There is no a fire')
    else:
        st.warning('Prediciton: There is a fire')
