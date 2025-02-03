import time
import streamlit as st
import pandas as pd
from functools import reduce
import scipy

from src.Preprocessing.utils import resample_and_apply_bandpass_filter


def extract_eeg_from_specific_file(file_name):
    """
    Load a single .mat file, extract its EEG and metadata.
    :param file_path: Path to .mat file
    :return: A dictionary containing EEG data and associated metadata.
    """
    mat_data = scipy.io.loadmat(file_name)
    key = next(key for key in mat_data.keys() if "segment" in key)
    segment = mat_data[key][0, 0]

    if "interictal" in file_name:
        segment_type = "interictal"
        labels = None
    elif "preictal" in file_name:
        segment_type = "preictal"
        labels = None
    else:
        segment_type = "test"
        labels = segment["labels"]

    # Extracting data and metadata from the wrapped nested structure

    eeg_data = segment["data"]
    if labels is not None:
        labels = labels[0, :].astype(int).tolist()
    data_length = int(segment["data_length_sec"][0, 0])
    sampling_frequency = float(segment["sampling_frequency"][0, 0])
    channels = segment["channels"]
    channels = channels.flatten().tolist()

    channels = [str(channel[0])[-4:] for channel in channels]

    return {
        "eeg_data": eeg_data,
        "labels": labels,
        "metadata": {
            "type": segment_type,
            "duration": data_length,
            "sampling_frequency": sampling_frequency,
            "channels": channels,
        },
    }


def display_metadata_in_expander(metadata, expander_title="EGG File Details"):
    """
    Display a dictionary in a table format inside a Streamlit expander.

    Args:
        data_dict (dict): The dictionary to display.
        expander_title (str): Title of the expander.
    """

    metadata_formatted = dict(metadata)

    duration = metadata["duration"] / 60  # Convert to minutes
    channels_list = metadata["channels"]

    metadata_formatted["duration"] = f"{duration:.0f} min"
    metadata_formatted["sampling_frequency"] = (
        f'{metadata["sampling_frequency"]:.2f} Hz'
    )
    metadata_formatted["channels"] = reduce(
        lambda acc, c: acc + ", " + c, channels_list
    )

    with st.sidebar.expander(expander_title):
        # Convert the dictionary into a DataFrame for display
        metadata_df = pd.DataFrame(
            metadata_formatted.items(), columns=["Information", "Value"]
        )
        metadata_df = metadata_df.set_index("Information")

        # Display the table
        st.table(metadata_df)


def simulate_streaming(data, chunk_size, update_period):
    """
    Simulates real-time streaming of data in chunks.

    This function takes a 2D array of data and yields consecutive chunks along the
    second axis to simulate a streaming process. Each chunk is delayed by the specified
    `update_period` to mimic real-time behavior.

    Args:
        data (np.ndarray): The input data, typically a 2D array where rows represent
                           signals (e.g., channels) and columns represent time points.
        chunk_size (int): The number of time points to include in each chunk.
        update_period (float): The time interval (in seconds) between yielding consecutive chunks.

    Yields:
        np.ndarray: A chunk of the data with shape (n_channels, chunk_size), where
                    `n_channels` is the number of rows in `data` and `chunk_size` is
                    the specified number of columns.
    """
    for i in range(0, data.shape[1], chunk_size):
        chunk = data[:, i : i + chunk_size]
        yield chunk
        time.sleep(update_period)


# Load CSS from external file
def load_css():
    with open("src/FrontEnd/style.css", "r", encoding="utf-8") as file:
        css = file.read()
    return css


def display_eeg_classes(labels, expander_title="EEG Classes Over Time"):
    """
    Display EEG classes over time in a Streamlit expander.

    Args:
        labels (list): List of labels with numbers 0 and 1. 0 corresponds to class "interictal" and 1 to "preictal".
        expander_title (str): Title of the expander.
    """
    time_periods = [f"{i*10}-{(i+1)*10} min" for i in range(len(labels))]
    classes = ["interictal" if label == 0 else "preictal" for label in labels]

    data = {"Time Period": time_periods, "EEG Class": classes}
    df = pd.DataFrame(data)

    with st.sidebar.expander(expander_title):
        st.table(df)


@st.cache_data
def load_and_prefilter_eeg(uploaded_file, new_sampling_freq):
    """
    Load EEG data from a specific file, resample it, and apply a bandpass filter.
    """
    data = extract_eeg_from_specific_file(uploaded_file)
    eeg_data = data["eeg_data"]
    metadata = data["metadata"]
    labels = data["labels"]

    total_duration = data["metadata"]["duration"]

    # Resample and filter data
    eeg_segment = {"eeg_data": eeg_data, "label": None}
    eeg_data, _ = resample_and_apply_bandpass_filter(
        eeg_segment, new_sampling_freq=new_sampling_freq, time_length=total_duration
    )

    metadata["sampling_frequency"] = (
        new_sampling_freq  # Update metadata after resampling
    )
    return eeg_data, metadata, labels
