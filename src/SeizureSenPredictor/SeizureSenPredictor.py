import numpy as np
import tensorflow as tf

from src.Model.MultiViewConvModelAttention import MultiViewConvModelWithAttention
from src.Model.utils import (
    load_config,
    load_model_from_config,
    load_scalers_from_config,
)
from src.Preprocessing.Feature import Feature
from src.Preprocessing.utils import transform_features_to_tensor, scale_across_time_tf


class SeizureSenPredictor:
    def __init__(self, model_config_path):
        self.config = load_config(model_config_path)

        # Load the model
        # Change the class model here if needed
        self.model = load_model_from_config(
            self.config, MultiViewConvModelWithAttention
        )

        self.scalers_time, self.scalers_freq = load_scalers_from_config(self.config)

    def classify_eeg(self, time_slices_list: list[np.ndarray]):
        """
        Classify EEG data as interictal or preictal.
        :param time_slices_list: A list of 30s time slices of EEG data.
        :return: A string indicating the class of the EEG data and the probability of the prediction.
        """

        X_time, X_freq = self._preprocess_eeg(time_slices_list)
        prob = 0.9

        return "interictal", prob

    def _preprocess_eeg(
        self, time_slices_list: list[np.ndarray]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess EEG data for the model. This includes feature extraction and scaling across time steps.
        :param time_slices_list: A list of 30s time slices of EEG data.
        :return: A tuple of tensors containing the preprocessed time and frequency domain features.
        """
        model_time_steps = self.config["model_time_steps"]

        if len(time_slices_list) != model_time_steps:
            raise ValueError(
                f"Expected {model_time_steps} time slices, but got {len(time_slices_list)}"
            )

        feature_preprocessor = Feature(400)

        X_time = (
            []
        )  # To store features in time domain for all slices. It is a list of dicts
        X_freq = (
            []
        )  # To store features in frequency domain for all slices. It is a list of dicts

        for time_slice in time_slices_list:
            eeg_slice_preprocessed = feature_preprocessor.time_domain_pca(time_slice)
            X_time.append(eeg_slice_preprocessed)

            eeg_slice_preprocessed = feature_preprocessor.freq_domain(time_slice)
            X_freq.append(eeg_slice_preprocessed)

        model_time_steps = self.config["model_time_steps"]

        # Convert the lists of dicts to tensors
        X_time = transform_features_to_tensor(
            X_time, model_time_steps
        )  # (n_samples, n_channels, time_features, steps)
        X_freq = transform_features_to_tensor(
            X_freq, model_time_steps
        )  # (n_samples, n_channels, freq_features, steps)

        # Scale the features across time. Each feature in each channel is scaled separetely
        X_time = scale_across_time_tf(X_time, self.scalers_time)
        X_freq = scale_across_time_tf(X_freq, self.scalers_freq)

        return X_time, X_freq

    def get_model_time_steps(self):
        """
        Get the number of time steps the model expects.
        """
        return self.config["model_time_steps"]
