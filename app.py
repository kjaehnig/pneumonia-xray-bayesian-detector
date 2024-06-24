import pickle as pk
import streamlit as st
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model


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
    num_classes = 2  # Normal and Pneumonia
    predicted_probabilities = np.empty(shape=(n_iter, num_classes))
    for i in tqdm(range(n_iter), leave=False):
        with st.spinner("Loading TensorFlow model..."):
            predicted_probabilities[i] = model(image[np.newaxis, :]).mean().numpy()[0]
    return predicted_probabilities

# Streamlit app
st.title("Pneumonia Detection Probability")
st.sidebar.title("Settings")

# Sidebar slider for number of predictions
n_iter = st.sidebar.slider("Number of Predictions", min_value=2, max_value=50, value=10)

# Dropdown menu for image selection
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.npz')]
selected_image_file = st.sidebar.selectbox("Select an Image", image_files)

predict_image = st.sidebar.button("Predict on this Xray")

if selected_image_file and predict_image:
    # Load and preprocess the image
    file_path = os.path.join(image_folder, selected_image_file)
    data = np.load(file_path)
    image = data['x']
    true_label = np.argmax(data['y']).astype('int')
    # img = tf.image.resize(image, [299, 299, 3])  # Adjust size if necessary

    # Make predictions
    predicted_probabilities = make_predictions(model, image, n_iter)

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
    st.image(
        image.reshape(299, 299, 3)/255.,
        caption=f'Selected Image: {pred_label}',
        use_column_width=True,
        clamp=True,
        channels='BGR'
    )

    # Plot probabilities
    fig, ax = plt.subplots()
    bar = ax.bar(np.arange(2), pct_97p5, color='red')
    bar[true_int].set_color('green')
    ax.bar(np.arange(2), pct_2p5 - 0.1, lw=3, color='white')
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

    st.write(f"Prediction Probabilities: Normal - {pct_50[0]:.2f}, Pneumonia - {pct_50[1]:.2f}")
