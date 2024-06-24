
import streamlit as st
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("path_to_your_model.h5", compile=False)

# Function to make predictions
def make_predictions(model, image, n_iter):
    num_classes = 2  # Normal and Pneumonia
    predicted_probabilities = np.empty(shape=(n_iter, num_classes))
    for i in tqdm(range(n_iter), leave=False):
        predicted_probabilities[i] = model(image[np.newaxis, :]).mean().numpy()[0]
    return predicted_probabilities

# Streamlit app
st.title("Pneumonia Detection Probability")
st.sidebar.title("Settings")

# Sidebar slider for number of predictions
n_iter = st.sidebar.slider("Number of Predictions", min_value=10, max_value=100, value=50)

# Dropdown menu for image selection
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.npz')]
selected_image_file = st.sidebar.selectbox("Select an Image", image_files)

if selected_image_file:
    # Load and preprocess the image
    file_path = os.path.join(image_folder, selected_image_file)
    data = np.load(file_path)
    image = data['image']
    image = tf.image.resize(image, [299, 299])  # Adjust size if necessary
    image = image / 255.0  # Normalize the image

    # Make predictions
    predicted_probabilities = make_predictions(model, image, n_iter)

    # Calculate percentiles
    pct_2p5 = np.percentile(predicted_probabilities, 2.5, axis=0)
    pct_97p5 = np.percentile(predicted_probabilities, 97.5, axis=0)
    pct_50 = np.percentile(predicted_probabilities, 50, axis=0)

    # Determine labels
    class_names = ['Normal', 'Pneumonia']
    pred_int = np.argmax(pct_50)
    true_int = pred_int  # This should be replaced with the actual label if known
    pred_label = class_names[pred_int]

    # Display results
    st.image(image, caption=f'Selected Image: {pred_label}', use_column_width=True)

    # Plot probabilities
    fig, ax = plt.subplots()
    bar = ax.bar(np.arange(2), pct_97p5, color='red')
    bar[true_int].set_color('green')
    ax.bar(np.arange(2), pct_2p5 - 0.02, lw=3, color='white')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title(f"Prediction: {pred_label}", color='green' if pred_int == true_int else 'red', fontweight='bold')

    st.pyplot(fig)

    st.write(f"Prediction Probabilities: Normal - {pct_50[0]:.2f}, Pneumonia - {pct_50[1]:.2f}")
