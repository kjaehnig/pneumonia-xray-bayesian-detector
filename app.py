import pickle as pk
import streamlit as st
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
import cv2

class MyBayesianModel:
    def __init__(self, model):
        self.model = model

    @tf.function
    def predict(self, img):
        """
        Perform inference using the trained Bayesian model.

        Args:
            input_data (tf.Tensor): Input data for prediction.

        Returns:
            tf.Tensor: Model predictions.
        """
        return self.model(img[np.newaxis, :]).mean().numpy()[0]


def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)


def divergence(q, p, _):
    return tfd.kl_divergence(q, p) / 112799.
# Load the trained model

custom_objs = {
    'neg_loglike': lambda x,y: -y.log_prob(x),
    'divergence': lambda q,p: trd.divergence(q, p) / 5216.
}
# model = load_model("trained_model", compile=True, custom_objects=custom_objs)

@st.cache_resource
def load_model_as_class_into_streamlit():
    tf.keras.backend.clear_session()
    with st.spinner("Loading TensorFlow model..."):
        from pneumonia_bcnn_detector import build_mdl

        @tf.function
        def neg_loglike(ytrue, ypred):
            return -ypred.log_prob(ytrue)

        @tf.function
        def divergence(q, p, _):
            return tfd.kl_divergence(q, p) / 6214.

        model = build_mdl()

        with open("trained_model/trained_model_weights.pk", 'rb') as whts:
            weights_pk = pk.load(whts)

        model.set_weights(weights_pk)

        # mdl_class = MyBayesianModel(model)

    return model

model = load_model_as_class_into_streamlit()

# Function to make predictions
def make_predictions(model, image, n_iter):
    progressbar = st.progress(0)

    num_classes = 2  # Normal and Pneumonia
    predicted_probabilities = np.empty(shape=(n_iter, num_classes))
    for i, per in enumerate(np.linspace(0, 100, n_iter).astype('int')):
        progressbar.progress(i/n_iter, text=f'predicting: {per:.2f}%')
        predicted_probabilities[i] = model(image[np.newaxis, :]).mean().numpy()[0]
    progressbar.empty()
    return predicted_probabilities

# Streamlit app
st.title("Bayesian Pneumonia Detection")
with st.expander("Description"):
    st.write("""
    This website hosts a neural network model that's trained to diagnose pneumonia from chest X-ray images. 
    Pneumonia is a significant health issue and causes many deaths amongst children under the age of 5. 
    This AI model uses a dataset of over 5,200 X-ray images of both normal and pneumonia cases.
    This model uses the EfficientNetV2S model that is pre-trained on 'ImageNet' to extract image features. It then uses 
    Bayesian neural network layers for classification. Bayesian models are advantageous as they quantify uncertainty 
    in predictions, which is crucial for medical diagnostics. Data augmentation techniques enhance the model's 
    robustness by randomly flipping an image, modifying the image contrast, and shifting the image around.
    The final model achieved an accuracy of 95.3%, which out performs the baseline model from the paper this dataset was
    sourced from. It effectively minimizes false negatives, ensuring pneumonia cases are not missed. 
    The Bayesian layers also provide confidence intervals for predictions, improving diagnostic reliability.
    This AI model demonstrates significant potential in assisting medical diagnostics by offering quick and 
    reliable pneumonia detection from chest X-rays, improving treatment outcomes and resource allocation in healthcare.
    """)
st.sidebar.title("Settings")

# Sidebar slider for number of predictions
n_iter = st.sidebar.slider("Number of Predictions", min_value=2, max_value=50, value=10)

# Dropdown menu for image selection
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.npz')]
image_files.sort()
selected_image_file = st.sidebar.selectbox("Select an Image", image_files)

select_img_size = st.sidebar.selectbox("Select image display size", ['small', 'medium', 'large'])

use_modified_img = st.sidebar.checkbox("Use modified Image")

alpha_val = st.sidebar.slider("Alpha (contrast)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
beta_val = st.sidebar.slider("Beta (brightness)", min_value=0, max_value=100, value=0, step=1)

flip_image_h = st.sidebar.checkbox("Flip Image (horizontal)")
flip_image_v = st.sidebar.checkbox("Flip Image (vertical)")

predict_image = st.sidebar.button("Predict!")

img_size = (299, 299)
if select_img_size == 'small':
    img_size = (299, 299)
if select_img_size == 'medium':
    img_size = (600, 600)
if select_img_size == 'large':
    img_size = (1200, 1200)

if selected_image_file and predict_image:
    # Load and preprocess the image
    file_path = os.path.join(image_folder, selected_image_file)
    data = np.load(file_path)
    image = data['x']
    true_label = np.argmax(data['y']).astype('int')
    # img = tf.image.resize(image, [299, 299, 3])  # Adjust size if necessary

    # Make predictions
    if use_modified_img:
        pred_image = np.zeros(image.shape, image.dtype)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    pred_image[y, x, c] = np.clip(alpha_val * image[y, x, c] + beta_val, 0, 255)
        if flip_image_h:
            pred_image = cv2.flip(pred_image, 1)
        if flip_image_v:
            pred_image = cv2.flip(pred_image, 0)
    else:
        pred_image = image
    predicted_probabilities = make_predictions(model, pred_image, n_iter)

    # Calculate percentiles
    pct_2p5 = np.percentile(predicted_probabilities, 2.5, axis=0)
    pct_97p5 = np.percentile(predicted_probabilities, 97.5, axis=0)
    pct_50 = np.percentile(predicted_probabilities, 50, axis=0)

    # Determine labels
    class_names = ['Normal', 'Pneumonia']
    pred_int = np.argmax(pct_50)
    true_int = true_label  # This should be replaced with the actual label if known
    pred_label = class_names[pred_int]

    # Display results
    if use_modified_img:
        img_cols = st.columns(2)
        img_cols[0].image(
            image/255.,
            caption=f'Selected Image: {class_names[true_int]} (original)',
            use_column_width=True,
            # width=img_size[0],
            clamp=True,
            channels='BGR'
        )
        img_cols[1].image(
            pred_image/255.,
            caption=f'Selected Image: {class_names[true_int]} (modified)',
            use_column_width=True,
            clamp=True,
            channels='BGR'
        )
    else:
        st.image(
            image/255.,
            caption=f'Selected Image: {class_names[true_int]}',
            # use_column_width=True,
            width=img_size[0],
            clamp=True,
            channels='BGR'
        )

    # Plot probabilities
    fig, ax = plt.subplots()
    bar = ax.bar(np.arange(2), pct_97p5, color='red')
    bar[true_int].set_color('green')
    ax.bar(np.arange(2), pct_2p5, lw=3, color='white', width=0.9)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability', fontweight='bold')
    # ax.set_xlabel(
    #     f"{'CORRECT' if true_int == pred_int else 'INCORRECT'} Prediction",
    #     color='green' if true_int == pred_int else 'red',
    #     fontweight='bold')
    ax.set_title(
        f"{'CORRECT' if true_int == pred_int else 'INCORRECT'} Prediction: {pred_label}",
        color='green' if pred_int == true_int else 'red',
        fontweight='bold')

    st.pyplot(fig)

    st.write(f"Prediction Probabilities: 50, (2.5, 97.5)")
    st.write(f"      Normal: {pct_50[0]:.2f} ({pct_2p5[0]:.2f}, {pct_97p5[0]:.2f})")
    st.write(f"      Pneumonia: {pct_50[1]:.2f} ({pct_2p5[1]:.2f}, {pct_97p5[1]:.2f})")
