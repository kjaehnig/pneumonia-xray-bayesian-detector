import pickle as pk
import streamlit as st
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
import cv2
import seaborn as sns

image_folder = 'images'

def load_image_file_names():
    # Dropdown menu for image selection
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.npz')]
    image_files.sort()

    true_labels = []
    for imgii in image_files:
        imgpath = os.path.join(image_folder, imgii)
        # print(imgpath)
        data = np.load(imgpath)
        true_labels.append('Pneumonia' if np.argmax(data['y']).astype(int)==1 else 'Normal')

    image_names = [tl+"_img_"+imf.split('_')[-1].split('.')[0] for tl, imf in list(zip(true_labels, image_files))]
    return image_names
 

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


# Load the trained model
@st.cache_resource
def load_model_as_class_into_streamlit():
    tf.keras.backend.clear_session()
    with st.spinner("Loading TensorFlow model..."):
        from pneumonia_bcnn_detector import build_mdl


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
        progressbar.progress(int(per), text=f'predicting: {per-i:.2f}%')
        predicted_probabilities[i] = model(image[np.newaxis, :]).mean().numpy()[0]
    progressbar.empty()
    return predicted_probabilities


def make_comparative_preds(model, images, n_iter):
    progressbar = st.progress(0)

    num_classes = 2  # Normal and Pneumonia
    predicted_probabilities = np.empty(shape=(n_iter, len(images), num_classes))
    for i, per in enumerate(np.linspace(0, 100, n_iter).astype('int')):
        progressbar.progress(int(per), text=f'predicting: {per-i:.2f}%')
        predicted_probabilities[i] = model(images).mean().numpy()
    progressbar.empty()
    return predicted_probabilities


def make_violin_fig(true_int, predicted_probabilities, class_names):
    predicted_probabilities = predicted_probabilities.reshape(st.session_state.n_iter, len(class_names))
    pct_2p5 = np.percentile(predicted_probabilities, 2.5, axis=0)
    pct_97p5 = np.percentile(predicted_probabilities, 97.5, axis=0)
    pct_50 = np.percentile(predicted_probabilities, 50, axis=0)

    pred_int = np.argmax(pct_50)
    pred_label = class_names[pred_int]

    fig, ax = plt.subplots()
    # bar = ax.bar(np.arange(2), pct_97p5, color='red')
    # bar[true_int].set_color('green')
    # ax.bar(np.arange(2), pct_2p5, lw=3, color='white', width=0.9)
    violin_colors = [
        'red' if true_int == 1 else 'green',
        'red' if true_int == 0 else 'green'
    ]

    sns.violinplot(
        predicted_probabilities,
        palette=violin_colors
    )
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

    return fig, (pct_2p5, pct_50, pct_97p5)

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

non_medical_warning = """
    :red[This app is entirely demonstrative and **SHOULD NOT** be used for any medical or diagnostic
    purposes.]
    """
with st.sidebar.expander("A Warning on usage."):
    st.markdown(non_medical_warning)


# st.markdown("""**Select image and hit Predict!**""")

image_names = load_image_file_names()
st.markdown("""**Step 1 - :red[Select an image]**""")
selected_image_file = st.selectbox('Image Selector', image_names)


st.markdown("""**Step 2 - :orange[Modify image (optional)]**""")
modifier_cols = st.columns(5)
modifier_cols[0].write("Flip Image")
flip_image_h = modifier_cols[1].checkbox("Horizontal")
flip_image_v = modifier_cols[2].checkbox("Vertical")
shot_switch = modifier_cols[3].checkbox("Seasoning")
use_modified_img = modifier_cols[4].checkbox("Use modified")

st.markdown("""**Step 3 - :green[Perform image classification]**""")
predict_image = st.button("Predict!")


st.sidebar.title(" Additional settings")

# Sidebar slider for number of predictions
n_iter = st.sidebar.slider("Number of Predictions", min_value=2, max_value=50, value=10, key='n_iter')



alpha_val = st.sidebar.slider("Alpha (contrast)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
beta_val = st.sidebar.slider("Beta (brightness)", min_value=0, max_value=100, value=0, step=1)

# shot_cols = st.sidebar.columns(2)
shot_noise = st.sidebar.slider("Shot Noise Factor", min_value=0.0, max_value=1.0, step=0.05)
# shot_switch = shot_cols[1].checkbox(" ")

if selected_image_file:
    class_names = ['Normal', 'Pneumonia']

    # Load and preprocess the image
    selected_img_path = f"chest_xray_testimg_{selected_image_file.split('_')[-1]}.npz"
    file_path = os.path.join(image_folder, selected_img_path)
    data = np.load(file_path)
    image = data['x']
    pred_image = image
    true_int = np.argmax(data['y']).astype('int')
    # img = tf.image.resize(image, [299, 299, 3])  # Adjust size if necessary
    img_cols = st.columns(2)

    img_cols[0].image(
        image / 255.,
        # caption=f'Selected Image: {class_names[true_int]} (original)',
        use_column_width=True,
        clamp=True,
        channels='BGR'
    )
    img_cols[0].markdown(f"""Chest X-Ray Type: {class_names[true_int]} (Original)""")
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
        if shot_switch:
            # imp_noise=np.zeros((299, 299, 3),dtype=image.dtype)
            # cv2.randu(imp_noise, 0, 255)
            # imp_noise=cv2.threshold(imp_noise, shot_noise, 255, cv2.THRESH_BINARY)[1]
            # pred_image = np.clip(cv2.add(pred_image, imp_noise), 0, 255)
            # pred_image = add_shot_noise(image, shot_noise)
            pred_image = np.clip(pred_image + shot_noise * np.random.poisson(pred_image), 0, 255)




        img_cols[1].image(
            pred_image/255.,
            # caption=f'Selected Image: {class_names[true_int]} (modified)',
            use_column_width=True,
            clamp=True,
            channels='BGR'
        )
        img_cols[1].markdown(f"""Chest X-Ray Type: {class_names[true_int]} :red[(Modified)]""")

    if predict_image:
        if use_modified_img:
            predicted_probabilities = make_comparative_preds(
                model, 
                np.array([image, pred_image]), 
                n_iter)
            

            fig, (pct_2p5, pct_50, pct_97p5) = make_violin_fig(
                true_int, 
                predicted_probabilities[:, 0, :], 
                class_names
            )

            plot_cols = st.columns(2)

            plot_cols[0].pyplot(fig)
            plot_cols[0].write(f"Prediction Probabilities: 50, (2.5, 97.5)")
            plot_cols[0].write(f"Normal: {pct_50[0]:.2f} ({pct_2p5[0]:.2f}, {pct_97p5[0]:.2f})")
            plot_cols[0].write(f"Pneumonia: {pct_50[1]:.2f} ({pct_2p5[1]:.2f}, {pct_97p5[1]:.2f})")


            fig, (pct_2p5, pct_50, pct_97p5) = make_violin_fig(
                true_int, 
                predicted_probabilities[:, 1, :], 
                class_names
            )

            plot_cols[1].pyplot(fig)
            plot_cols[1].write(f"Prediction Probabilities: 50, (2.5, 97.5)")
            plot_cols[1].write(f"Normal: {pct_50[0]:.2f} ({pct_2p5[0]:.2f}, {pct_97p5[0]:.2f})")
            plot_cols[1].write(f"Pneumonia: {pct_50[1]:.2f} ({pct_2p5[1]:.2f}, {pct_97p5[1]:.2f})")
        else:
            predicted_probabilities = make_predictions(model, pred_image, n_iter)

            fig, (pct_2p5, pct_50, pct_97p5) = make_violin_fig(
                true_int, 
                predicted_probabilities, 
                class_names
            )

            st.pyplot(fig)
            st.write(f"Prediction Probabilities: 50, (2.5, 97.5)")
            st.write(f"Normal: {pct_50[0]:.2f} ({pct_2p5[0]:.2f}, {pct_97p5[0]:.2f})")
            st.write(f"Pneumonia: {pct_50[1]:.2f} ({pct_2p5[1]:.2f}, {pct_97p5[1]:.2f})")
        # Calculate percentiles


        # Determine labels


        # Plot probabilities
        # fig, ax = plt.subplots()
        # # bar = ax.bar(np.arange(2), pct_97p5, color='red')
        # # bar[true_int].set_color('green')
        # # ax.bar(np.arange(2), pct_2p5, lw=3, color='white', width=0.9)
        # violin_colors = [
        #     'red' if true_int == 1 else 'green',
        #     'red' if true_int == 0 else 'green'
        # ]

        # sns.violinplot(
        #     predicted_probabilities[0],
        #     palette=violin_colors
        # )
        # ax.set_xticks(np.arange(2))
        # ax.set_xticklabels(class_names)
        # ax.set_ylim([0, 1])
        # ax.set_ylabel('Probability', fontweight='bold')
        # # ax.set_xlabel(
        # #     f"{'CORRECT' if true_int == pred_int else 'INCORRECT'} Prediction",
        # #     color='green' if true_int == pred_int else 'red',
        # #     fontweight='bold')
        # ax.set_title(
        #     f"{'CORRECT' if true_int == pred_int else 'INCORRECT'} Prediction: {pred_label}",
        #     color='green' if pred_int == true_int else 'red',
        #     fontweight='bold')


