import tensorflow as tf
from keras.api import backend as K


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for binary classification.

    Parameters:
        gamma (float): Focusing parameter. Higher values focus on hard-to-classify examples.
        alpha (float): Balancing factor for class imbalance.

    Returns:
        A function to be used as a loss in model.compile().
    """

    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Use TensorFlow's clip_by_value

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))  # For binary classification

    return focal_loss_fixed



def f1_loss(y_true, y_pred):
    """
    F1 Loss for binary classification.
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.round(y_pred)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return 1 - tf.reduce_mean(f1)