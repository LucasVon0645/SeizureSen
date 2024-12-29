import tensorflow as tf
from keras.layers import Layer


class StandardScalerLayer(Layer):
    def __init__(self, means, stds, **kwargs):
        """
        Custom scaling layer that stores mean and standard deviation for each feature.

        Parameters:
            means (np.ndarray): Mean values for each channel and
                                feature, shape (n_channels, n_features).
            stds (np.ndarray): Standard deviation values, shape (n_channels, n_features).
        """
        super(StandardScalerLayer, self).__init__(**kwargs)
        self.means = tf.constant(means, dtype=tf.float32)
        self.stds = tf.constant(stds, dtype=tf.float32)

    def call(self, inputs):
        """
        Scale inputs using stored mean and standard deviation.

        Parameters:
            inputs (tf.Tensor): Input tensor, shape (n_samples, n_channels, n_features, steps).

        Returns:
            tf.Tensor: Scaled inputs.
        """
        return (inputs - self.means[:, :, tf.newaxis]) / self.stds[:, :, tf.newaxis]
