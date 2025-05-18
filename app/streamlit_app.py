import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load models once
@st.cache_resource
def load_models():
    unet = tf.keras.models.load_model('saved_models/unet_model.h5', compile=False)
    # Also load your classifier here if needed
    return unet

unet = load_models()

st.title("Brain Tumor Segmentation Demo")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=['jpg','jpeg','png','tif','tiff'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((256, 256))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred_mask = unet.predict(img_input)[0,:,:,0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    ax[1].imshow(img)
    ax[1].imshow(pred_mask, cmap='Reds', alpha=0.5)
    ax[1].set_title("Predicted Tumor Mask")
    ax[1].axis('off')

    st.pyplot(fig)
