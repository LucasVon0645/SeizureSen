import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.DataExtraction.DataExtractor import DataExtractor
from src.Preprocessing.utils import (
    eeg_slices,
    divide_into_frequency_chunks,
    calculate_band_features,
)


data_dir = "data"
test_labels_file = "TestLabels.csv"

data_extractor = DataExtractor(
    data_directory=data_dir, test_labels_file=test_labels_file
)
data_extractor.load_data(
    dog_ids=["Dog_1"], segment_types=["interictal", "preictal", "test"]
)
loaded_data = data_extractor.get_data()

interictal_segments = loaded_data["interictal"]
metadata = data_extractor.get_metadata()
print("Metadata:", metadata)
sampling_freq = metadata["sampling_frequency"]
window_duration = 30

preictal_segments = loaded_data["preictal"]

if interictal_segments:
    first_interictal_segment = interictal_segments[0]["eeg_data"]
    print(f"Original Interictal Segment Shape: {first_interictal_segment.shape}")

    # Apply eeg_slices
    slices = eeg_slices(first_interictal_segment, sampling_freq, window_duration)
    print(f"Number of slices: {len(slices)}")

    interictal_band_features = []  # To store band features for all slices

    for idx, slice_segment in enumerate(slices):
        print(f"Slice {idx + 1} Shape: {slice_segment.shape}")

        # Divide into frequency bands
        frequency_band_data = divide_into_frequency_chunks(slice_segment, sampling_freq)
        print(f"Frequency bands divided for Slice {idx + 1}.")

        # Calculate mean log amplitude and standard deviation
        band_features = calculate_band_features(frequency_band_data)
        print(f"Band features calculated for Slice {idx + 1}.")

        interictal_band_features.append(band_features)

    print("\nInterictal frequency band processing and feature extraction complete.")

if preictal_segments:
    first_preictal_segment = preictal_segments[0]["eeg_data"]
    print(f"Original Preictal Segment Shape: {first_preictal_segment.shape}")

    slices = eeg_slices(first_preictal_segment, sampling_freq, window_duration)
    print(f"Number of slices: {len(slices)}")

    preictal_band_features = []

    for idx, slice_segment in enumerate(slices):
        print(f"Slice {idx + 1} Shape: {slice_segment.shape}")

        frequency_band_data = divide_into_frequency_chunks(slice_segment, sampling_freq)
        print(f"Frequency bands divided for Slice {idx + 1}.")

        band_features = calculate_band_features(frequency_band_data)
        print(f"Band features calculated for Slice {idx + 1}.")

        preictal_band_features.append(band_features)

    print("\nPreictal frequency band processing and feature extraction complete.")


def extract_alpha_means(band_features_list):
    return [features["alpha_mean"] for features in band_features_list]


# Extract alpha band features for interictal and preictal data
interictal_alpha_means = extract_alpha_means(interictal_band_features)
preictal_alpha_means = extract_alpha_means(preictal_band_features)


# Plot alpha band features for interictal and preictal data across all 16 channels
def plot_alpha_band_features(interictal_means, preictal_means):

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(interictal_means)),
        interictal_means,
        label="Interictal - Alpha Mean",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        range(len(preictal_means)),
        preictal_means,
        label="Preictal - Alpha Mean",
        linestyle="-",
        marker="x",
    )

    plt.xlabel("Slice Index")
    plt.ylabel("Alpha Band Mean Log Amplitude")
    plt.title("Alpha Band Features for Interictal and Preictal Segments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot the alpha band features
plot_alpha_band_features(interictal_alpha_means, preictal_alpha_means)
