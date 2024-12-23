import numpy as np

from sklearn.decomposition import PCA
from src.Preprocessing.utils import (
    divide_into_frequency_chunks,
    calculate_band_features,
)


class Feature:
    def __init__(self, sampling_freq):
        self.sampling_freq = sampling_freq

    def time_domain_pca(self, eeg_slice: np.ndarray, pca_components=16):
        """
        Perform PCA on a slice of data and return a 16x16 matrix.

        Parameters:
        - slice_data (numpy.ndarray): A 2D array with shape (16, slice_length),
        where 16 is the number of channels.
        - pca_components (int): Number of principal components to use for PCA.

        Returns:
        - pca_matrix (numpy.ndarray): A 16x16 matrix after applying PCA.
        """
        # Perform PCA on the slice data with principal components = 16
        pca = PCA(n_components=pca_components)
        # Fit and transform the slice data
        pca_result = pca.fit_transform(eeg_slice)
        # Log transformation for stability
        pca_matrix = np.log10(np.abs(pca_result))
        return pca_matrix

    def freq_domain(self, eeg_slice):
        """
        Extracts frequency domain features from an EEG slice.
        This method divides the given EEG slice into frequency bands and calculates
        the mean log amplitude and standard deviation for each band.
        Parameters:
        eeg_slice (numpy.ndarray): A 1D array representing a slice of EEG data.
        Returns:
        dict: A dictionary containing the calculated features for each frequency band.
        """
        # Ensure the input is a 2D array
        if eeg_slice.ndim != 2:
            raise ValueError("eeg_slice must be a 2D array with shape (channels, time_samples)")
        
        # Divide into frequency bands
        frequency_band_data = divide_into_frequency_chunks(
            eeg_slice, self.sampling_freq
        )

        # Calculate mean log amplitude and standard deviation
        band_features = calculate_band_features(frequency_band_data)

        return band_features
