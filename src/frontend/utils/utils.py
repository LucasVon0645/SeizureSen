import streamlit as st
import pandas as pd
from functools import reduce
import scipy


def extract_eeg_from_specific_file(file):
    """
    Load a single .mat file, extract its EEG and metadata.
    :param file_path: Path to .mat file
    :return: A dictionary containing EEG data and associated metadata.
    """
    mat_data = scipy.io.loadmat(file)
    key = next(key for key in mat_data.keys() if "segment" in key)
    segment = mat_data[key][0, 0]

    if "interictal" in file.name:
        segment_type = "interictal"
    elif "preictal" in file.name:
        segment_type = "preictal"
    else:
        segment_type = "test"

    # Extracting data and metadata from the wrapped nested structure

    eeg_data = segment["data"]
    data_length = int(segment["data_length_sec"][0, 0])
    sampling_frequency = float(segment["sampling_frequency"][0, 0])
    channels = segment["channels"]
    channels = channels.flatten().tolist()
    
    channels = [str(channel[0])[-4:] for channel in channels]

    return {
        "eeg_data": eeg_data,
        "metadata": {
            "type": segment_type,
            "duration": data_length,
            "sampling_frequency": sampling_frequency,
            "channels": channels
            }
    }

def display_metadata_in_expander(metadata, expander_title="EGG File Details"):
    """
    Display a dictionary in a table format inside a Streamlit expander.

    Args:
        data_dict (dict): The dictionary to display.
        expander_title (str): Title of the expander.
    """

    metadata_formatted = dict(metadata)

    duration = metadata["duration"]
    channels_list = metadata["channels"]

    metadata_formatted["duration"] = f"{duration} seconds"
    metadata_formatted["sampling_frequency"] = f'{metadata["sampling_frequency"]:.2f} Hz'
    metadata_formatted["channels"] = reduce(lambda acc, c: acc + ", " + c, channels_list)

    with st.expander(expander_title):
        # Convert the dictionary into a DataFrame for display
        metadata_df = pd.DataFrame(
            metadata_formatted.items(), columns=["Information", "Value"]
        )
        metadata_df = metadata_df.set_index("Information")

        # Display the table
        st.table(metadata_df)
