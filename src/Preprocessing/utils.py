import numpy as np
import os
from src.DataExtraction.DataExtractor import DataExtractor


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



"""data_dir = "data"
test_labels_file = "TestLabels.csv"

data_extractor = DataExtractor(data_directory=data_dir, test_labels_file=test_labels_file)
data_extractor.load_data(dog_ids=["Dog_1", "Dog_2"], segment_types=["interictal", "preictal", "test"])
loaded_data = data_extractor.get_data()

interictal_segments = loaded_data["interictal"]
metadata = data_extractor.get_metadata()
print("Metadata:", metadata)
sampling_freq = metadata["sampling_frequency"]
window_duration = 30

if interictal_segments:
    first_interictal_segment = interictal_segments[0]["eeg_data"]
    print(f"Original Interictal Segment Shape: {first_interictal_segment.shape}")

    # Apply eeg_slices
    slices = eeg_slices(first_interictal_segment, sampling_freq, window_duration)
    print(f"Number of slices: {len(slices)}")
    for idx, slice_segment in enumerate(slices):
        print(f"Slice {idx + 1} Shape: {slice_segment.shape}")"""