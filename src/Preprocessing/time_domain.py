import numpy as np
from sklearn.decomposition import PCA
from src.DataExtraction.DataExtractor import DataExtractor
from src.Preprocessing.utils import eeg_slices

def compute_pca(slice_data):
    """
    Perform PCA on a slice of data and return a 16x16 matrix.

    Parameters:
    - slice_data (numpy.ndarray): A 2D array with shape (16, slice_length),
      where 16 is the number of channels.

    Returns:
    - pca_matrix (numpy.ndarray): A 16x16 matrix after applying PCA.
    """
    pca = PCA(n_components=16)
    # Transpose slice_data to shape (slice_length, 16) for PCA
    pca_result = pca.fit_transform(slice_data)
    # Return the transformed data with shape (16, 16)
    pca_matrix = np.log10(np.abs(pca_result))  # Log transformation for stability
    return pca_matrix


# Example usage:
if __name__ == "__main__":
    data_dir = "data"
    test_labels_file = "TestLabels.csv"

    data_extractor = DataExtractor(data_directory=data_dir, test_labels_file=test_labels_file)
    data_extractor.load_data(dog_ids=["Dog_1"], segment_types=["interictal", "preictal", "test"])
    loaded_data = data_extractor.get_data()

    interictal_segments = loaded_data["interictal"]
    metadata = data_extractor.get_metadata()
    print("Metadata:", metadata)
    sampling_freq = metadata["sampling_frequency"]
    window_duration = 30

    # preictal_segments = loaded_data["preictal"]

    if interictal_segments:
        first_interictal_segment = interictal_segments[0]["eeg_data"]
        print(f"Original Interictal Segment Shape: {first_interictal_segment.shape}")

        # Apply eeg_slices
        slices = eeg_slices(first_interictal_segment, sampling_freq, window_duration)
        print(f"Number of slices: {len(slices)}")

        interictal_band_features = []  # To store band features for all slices

        for idx, slice_segment in enumerate(slices):
            print(f"Slice {idx + 1} Shape: {slice_segment.shape}")
            pca_result = compute_pca(slice_segment)
            print(f"PCA result shape for Slice {idx + 1}: {pca_result.shape}")
