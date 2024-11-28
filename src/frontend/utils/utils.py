import streamlit as st
import pandas as pd
from functools import reduce

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
