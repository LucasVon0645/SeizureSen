import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.DataExtraction.DataExtractor import DataExtractor
from src.Preprocessing.utils import (
    eeg_slices,
    save_preprocessed_data,
    resample_and_apply_bandpass_filter,
)
from src.Preprocessing.Feature import Feature

# In this script, we preprocess the EEG data to extract features for training and testing the model.
# We extract features in both time and frequency domains.
# Only data from Dog_1 is preprocessed in this script.

print("Preprocessing EEG data...")

DIR_RAW_DATA = "data"
TEST_LABELS_FILE = "TestLabels.csv"

data_extractor = DataExtractor(
    data_directory=DIR_RAW_DATA, test_labels_file=TEST_LABELS_FILE
)
data_extractor.load_data(
    dog_ids=["Dog_1"], segment_types=["interictal", "preictal", "test"]
)
loaded_data = data_extractor.get_data()

metadata = data_extractor.get_metadata()
print("Metadata: ", metadata)

SAMPLING_FREQ = 400  # New sampling frequency of EEG data in Hz
DIR_PREPROCESSED_DATA = "data/preprocessed/Dog_1"
SEGMENT_DURATION = 600  # Length of each segment in seconds (10 minutes)
WINDOW_DURATION = 30  # Duration of each slice in seconds

train_segments = loaded_data["preictal"] + loaded_data["interictal"]
test_segments = loaded_data["test"]

# Resample and apply bandpass filter to EEG data
train_segments_filtered = []
for eeg_segment in tqdm(train_segments, desc="Filtering train segments"):
    train_segments_filtered.append(
        resample_and_apply_bandpass_filter(
            eeg_segment, new_sampling_freq=SAMPLING_FREQ, time_length=SEGMENT_DURATION
        )
    )
train_segments = train_segments_filtered

test_segments_filtered = []
for eeg_segment in tqdm(test_segments, desc="Filtering test segments"):
    test_segments_filtered.append(
        resample_and_apply_bandpass_filter(
            eeg_segment, new_sampling_freq=SAMPLING_FREQ, time_length=SEGMENT_DURATION
        )
    )
test_segments = test_segments_filtered

train_slices_with_label = (
    []
)  # To store slices for all training segments along with their labels
test_slices_with_label = (
    []
)  # To store slices for all test segments along with their labels

# Slice the 10-minute segments into 30-second slices. A label is assigned to each slice.
for eeg_data_segment, label in tqdm(train_segments, desc="Slicing train segments"):
    slices = eeg_slices(eeg_data_segment, SAMPLING_FREQ, WINDOW_DURATION)
    train_slices_with_label.extend([(eeg_slice, label) for eeg_slice in slices])

for eeg_data_segment, label in tqdm(test_segments, desc="Slicing test segments"):
    slices = eeg_slices(eeg_data_segment, SAMPLING_FREQ, WINDOW_DURATION)
    test_slices_with_label.extend([(eeg_slice, label) for eeg_slice in slices])

print(f"Number of slices for training: {len(train_slices_with_label)}")
print(f"Number of slices for testing: {len(test_slices_with_label)}")

feature_preprocessor = Feature(SAMPLING_FREQ)

print("Preprocessing Training Dataset in Time Domain:")

X_time_train = []  # To store features for all slices
y_time_train = []  # To store labels for all slices

for eeg_slice, label in tqdm(train_slices_with_label):
    eeg_preprocessed = feature_preprocessor.time_domain_pca(eeg_slice)
    X_time_train.append(eeg_preprocessed)
    y_time_train.append(label)

X_time_train = np.array(X_time_train)
y_time_train = np.array(y_time_train)

print("Shape of X_time_train:", X_time_train.shape)
print("Shape of y_time_train:", y_time_train.shape)

print("Saving preprocessed training data in time domain...")
save_preprocessed_data(
    directory="data/preprocessed/Dog_1",
    filename="time_domain_train.npz",
    X=X_time_train,
    y=y_time_train,
)

print("Preprocessing Training Dataset in Frequency Domain:")

X_freq_train = []  # To store features for all slices
y_freq_train = []  # To store labels for all slices

for eeg_slice, label in tqdm(train_slices_with_label):
    eeg_preprocessed = feature_preprocessor.freq_domain(eeg_slice)
    X_freq_train.append(eeg_preprocessed)
    y_freq_train.append(label)

X_freq_train = np.array(X_freq_train)
y_freq_train = np.array(y_freq_train)

print("Shape of X_freq_train:", X_freq_train.shape)
print("Shape of y_freq_train:", y_freq_train.shape)

print("Saving preprocessed training data in frequency domain...")
save_preprocessed_data(
    directory="data/preprocessed/Dog_1",
    filename="freq_domain_train.npz",
    X=X_freq_train,
    y=y_freq_train,
)

print("Preprocessing Test Dataset in Time Domain:")

X_time_test = []  # To store features for all slices
y_time_test = []  # To store labels for all slices

for eeg_slice, label in tqdm(test_slices_with_label):
    eeg_preprocessed = feature_preprocessor.time_domain_pca(eeg_slice)
    X_time_test.append(eeg_preprocessed)
    y_time_test.append(label)

X_time_test = np.array(X_time_test)
y_time_test = np.array(y_time_test)

print("Shape of X_time_test:", X_time_test.shape)
print("Shape of y_time_test:", y_time_test.shape)

print("Saving preprocessed test data in time domain...")
save_preprocessed_data(
    directory="data/preprocessed/Dog_1",
    filename="time_domain_test.npz",
    X=X_time_test,
    y=y_time_test,
)

print("Preprocessing Test Dataset in Frequency Domain:")

X_freq_test = []  # To store features for all slices
y_freq_test = []  # To store labels for all slices

for eeg_slice, label in tqdm(test_slices_with_label):
    eeg_preprocessed = feature_preprocessor.freq_domain(eeg_slice)
    X_freq_test.append(eeg_preprocessed)
    y_freq_test.append(label)

X_freq_test = np.array(X_freq_test)
y_freq_test = np.array(y_freq_test)

print("Shape of X_freq_test:", X_freq_test.shape)
print("Shape of y_freq_test:", y_freq_test.shape)

print("Saving preprocessed test data in frequency domain...")
save_preprocessed_data(
    directory="data/preprocessed/Dog_1",
    filename="freq_domain_test.npz",
    X=X_freq_test,
    y=y_freq_test,
)

print("Preprocessing completed.")