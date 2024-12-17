import numpy as np
import os
from src.DataExtraction.DataExtractor import DataExtractor
import matplotlib.pyplot as plt



def eeg_slices(eeg_data, sampling_freq, window_duration):
    slice_samples = int(sampling_freq*window_duration) #no. of samples per slice
    total_samples = eeg_data.shape[1] #total no. of samples
    slices = []
    for i in range(0,total_samples,slice_samples):
        if i+slice_samples <= total_samples: #skipping the last slice if it doesn't fit in 30s window
            slices.append(eeg_data[:,i:i+slice_samples])
    return slices


def save_preprocessed_data(data_directory, file_name="preprocessed_data.npz", data=None):
    if data is None:
        raise ValueError("No data provided to save.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.normpath(os.path.join(script_dir, "..", "..", data_directory))
    file_path = os.path.join(save_dir, file_name)

    # Save data
    np.savez(file_path, **data)
    print(f"Data saved to {file_path}")


def load_preprocessed_data(data_directory, file_name="preprocessed_data.npz"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.normpath(os.path.join(script_dir, "..", "..", data_directory))
    file_path = os.path.join(load_dir, file_name)

    #Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    #Load data
    data = np.load(file_path, allow_pickle=True)
    print(f"Data loaded from {file_path}")
    return {key: data[key] for key in data}

def divide_into_frequency_chunks(eeg_data, sampling_freq):
    frequency_bands = {
        "delta": (0.1, 4),  # Delta: Deep sleep, restorative sleep and unconscious brain activity.
        "theta": (4, 8),  # Theta: Relaxation, light sleep, drowsiness and meditation, often linked to creativity.
        "alpha": (8, 12),  # Alpha: Calm wakefulness, relaxed state and idle mental processes, associated with resting but alert.
        "beta": (12, 30),  # Beta: Active thinking, problem-solving, decision-making and focused mental activity.
        "low_gamma": (30, 50),  # Low Gamma: Higher cognitive functions, learning and memory processing.
        "mid_gamma": (50, 70),  # Mid Gamma: Advanced problem-solving, information processing and heightened perception.
        "high_gamma_1": (70, 100),  # High Gamma 1: Enhanced brain activity during tasks requiring attention or conscious focus.
        "high_gamma_2": (100, 180)  # High Gamma 2: Intense brain activity during high-level processing and cognitive tasks.
    }

    num_samples = eeg_data.shape[1]
    freqs = np.fft.rfftfreq(num_samples, d=1/sampling_freq) # Frequency bins
    fft_magnitude = np.abs(np.fft.rfft(eeg_data, axis=1)) #FFT Magnitude

    frequency_band_data = {}

    for band_name, (low, high) in frequency_bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_data = fft_magnitude[:, band_mask]  # FFT magnitudes in the frequency band
        frequency_band_data[band_name] = band_data

    return frequency_band_data


def calculate_band_features(frequency_band_data):

    band_features = {} # Band features are the mean log amplitude and standard deviation.
    # This function returns a dictionary containing band features for each band

    for band_name, band_data in frequency_band_data.items():
        log_amplitude = np.log1p(band_data)  # log(1 + amplitude)

        mean_log_amplitude = np.mean(log_amplitude, axis=1)  # Mean for each channel
        std_log_amplitude = np.std(log_amplitude, axis=1)  # Standard deviation for each channel

        band_features[f"{band_name}_mean"] = mean_log_amplitude
        band_features[f"{band_name}_std"] = std_log_amplitude

    return band_features


"""
data_dir = "data"
test_labels_file = "TestLabels.csv"

data_extractor = DataExtractor(data_directory=data_dir, test_labels_file=test_labels_file)
data_extractor.load_data(dog_ids=["Dog_1", "Dog_2"], segment_types=["interictal", "preictal", "test"])
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
    plt.plot(range(len(interictal_means)), interictal_means, label="Interictal - Alpha Mean", linestyle='-', marker='o')
    plt.plot(range(len(preictal_means)), preictal_means, label="Preictal - Alpha Mean", linestyle='-', marker='x')

    plt.xlabel("Slice Index")
    plt.ylabel("Alpha Band Mean Log Amplitude")
    plt.title("Alpha Band Features for Interictal and Preictal Segments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the alpha band features
plot_alpha_band_features(interictal_alpha_means, preictal_alpha_means)
"""