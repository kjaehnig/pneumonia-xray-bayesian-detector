import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, GlobalAveragePooling2D)
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Metric

# class F1Score(Metric):
#     """
#     Custom F1 Score metric for evaluating model performance.
#
#     This metric calculates the F1 Score, which is the harmonic mean of precision and recall.
#     It is particularly useful for classification tasks with imbalanced classes.
#
#     Attributes:
#         true_positives (tf.Variable): Counter for true positive predictions.
#         false_positives (tf.Variable): Counter for false positive predictions.
#         false_negatives (tf.Variable): Counter for false negative predictions.
#     """
#     def __init__(self, name='f1_score', **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name='tp', initializer='zeros')
#         self.false_positives = self.add_weight(name='fp', initializer='zeros')
#         self.false_negatives = self.add_weight(name='fn', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         """
#         Updates the state of the metric with new predictions and true labels.
#
#         Args:
#             y_true (tf.Tensor): True labels.
#             y_pred (tf.Tensor): Predicted labels.
#             sample_weight (Optional[tf.Tensor]): Sample weights. Defaults to None.
#         """
#         y_true = tf.cast(y_true, tf.bool)
#         y_pred = tf.cast(tf.round(y_pred), tf.bool)
#         true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
#         false_positives = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
#         false_negatives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
#
#         self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_positives, self.dtype)))
#         self.false_positives.assign_add(tf.reduce_sum(tf.cast(false_positives, self.dtype)))
#         self.false_negatives.assign_add(tf.reduce_sum(tf.cast(false_negatives, self.dtype)))
#
#     def result(self):
#         """
#         Computes and returns the F1 Score.
#
#         Returns:
#             tf.Tensor: The F1 Score.
#         """
#         precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
#         recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
#         f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
#         return f1_score
#
#     def reset_state(self):
#         """Resets all of the metric state variables."""
#         self.true_positives.assign(0)
#         self.false_positives.assign(0)
#         self.false_negatives.assign(0)

tfd = tfp.distributions
tfpl = tfp.layers

def build_mdl():
    """
    Builds and compiles a Keras model with EfficientNetV2S as the base model.

    The model includes data augmentation layers, dense layers with L2 regularization,
    and dropout layers. The AdamW optimizer is used for compilation with custom learning
    rate and beta values.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """

    def neg_loglike(ytrue, ypred):
        return -ypred.log_prob(ytrue)

    def divergence(q, p, _):
        return tfd.kl_divergence(q, p) / 6214.

    K.clear_session()
    mdl = Sequential()

    # Fixed hyperparameters
    rdm_contrast = 0.32
    rdm_trans = 0.33
    rdm_flip = 'horizontal'
    l2_reg = 2e-06
    d1units = 256
    d2units = 256
    dropout_rate1 = 0.59
    dropout_rate2 = 0.68
    learning_rate = 4.5e-05
    beta_1 = 0.937
    beta_2 = 0.916
    weight_decay = 0.002

    # Data augmentation layers
    mdl.add(tf.keras.layers.RandomFlip(rdm_flip, input_shape=(299, 299, 3)))
    mdl.add(tf.keras.layers.RandomContrast(rdm_contrast))
    mdl.add(tf.keras.layers.RandomTranslation(rdm_trans, rdm_trans))

    # Load EfficientNetV2S as base model
    base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    mdl.add(base_model)
    mdl.add(GlobalAveragePooling2D())

    # L2 regularization
    kernel_regs = l2(l2_reg)

    # Dense layers with regularization and dropout
    mdl.add(tfpl.DenseReparameterization(
        units=d1units, activation='linear',
        activity_regularizer=kernel_regs,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence,
        )
    )
    mdl.add(Activation('relu'))
    mdl.add(Dropout(dropout_rate1))
    mdl.add(tfpl.DenseReparameterization(
        units=d2units, activation='linear',
        activity_regularizer=kernel_regs,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence,
        )
    )
    mdl.add(Activation('relu'))
    mdl.add(Dropout(dropout_rate2))

    mdl.add(tfpl.DenseReparameterization(
        units=tfpl.OneHotCategorical.params_size(2),
        activity_regularizer=kernel_regs,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence
        ),
    )


    mdl.add(tfpl.OneHotCategorical(
        2,
        convert_to_tensor_fn=tfd.Distribution.mode
        )
    )
    # Compile the model with AdamW optimizer
    optimizer = AdamW(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        weight_decay=weight_decay
    )

    mdl.compile(optimizer=optimizer, loss=neg_loglike, metrics=['accuracy'])

    return mdl
