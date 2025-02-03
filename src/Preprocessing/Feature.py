import numpy as np

from sklearn.decomposition import PCA
from src.Preprocessing.utils import (
    divide_into_frequency_chunks,
    calculate_band_features,
)


class Feature:
    """A class used to represent and extract features from EEG data.
    Attributes:
        sampling_freq (float): The sampling frequency of the EEG data.
    Methods:
        time_domain_pca(eeg_slice: np.ndarray, pca_components=16) -> dict:
            Perform PCA on a slice of EEG data and return a dictionary with principal components.
        freq_domain(eeg_slice: np.ndarray) -> dict:
            Extract frequency domain features from an EEG slice and return a dictionary with
            the calculated features.
    """

    def __init__(self, sampling_freq):
        self.sampling_freq = sampling_freq

    def time_domain_pca(self, eeg_slice: np.ndarray, pca_components=16):
        """
        Perform PCA on a slice of data and return a dictionary with keys as "pc1", "pc2", etc.

        Parameters:
        - slice_data (numpy.ndarray): A 2D array with shape (channels, time_samples),
        where 16 is the number of channels.
        - pca_components (int): Number of principal components to use for PCA.

        Returns:
        - pca_dict (dict): A dictionary with keys as "pc1", "pc2", etc. and values as arrays
        containing the numbers for the channels.
        """

        # Ensure the input is a 2D array
        if eeg_slice.ndim != 2:
            raise ValueError(
                "eeg_slice must be a 2D array with shape (channels, time_samples)"
            )

        # Perform PCA on the slice data with principal components = 16
        pca = PCA(n_components=pca_components)
        # Fit and transform the slice data
        pca_result = pca.fit_transform(eeg_slice)
        # Log transformation for stability
        pca_result = np.log10(np.abs(pca_result))

        # Create dictionary with keys as "pc1", "pc2", etc.
        pca_dict = {f"pc{i+1}": pca_result[:, i] for i in range(pca_components)}

        return pca_dict

    def freq_domain(self, eeg_slice, use_std_in_time_domain=False):
        """
        Extracts frequency domain features from an EEG slice.
        This method divides the given EEG slice into frequency bands and calculates
        the mean log amplitude and standard deviation for each band.
        Parameters:
        eeg_slice (numpy.ndarray): A 2D array representing a slice of EEG data (channels, time_samples).
        Returns:
        dict: A dictionary containing the calculated features for each frequency band.
        """
        # Ensure the input is a 2D array
        if eeg_slice.ndim != 2:
            raise ValueError(
                "eeg_slice must be a 2D array with shape (channels, time_samples)"
            )

        # Divide into frequency bands
        frequency_band_data = divide_into_frequency_chunks(
            eeg_slice, self.sampling_freq
        )
        
        if use_std_in_time_domain:
            # Calculate mean log amplitude and standard deviation
            band_features = calculate_band_features(frequency_band_data, use_std_in_time_domain, eeg_slice)
            return band_features
        else:
            # Calculate mean log amplitude and standard deviation
            band_features = calculate_band_features(frequency_band_data)
            return band_features
