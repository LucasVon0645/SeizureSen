import streamlit as st
import pandas as pd

def display_metadata_in_expander(metadata, expander_title="EGG File Details"):
    """
    Display a dictionary in a table format inside a Streamlit expander.

    Args:
        data_dict (dict): The dictionary to display.
        expander_title (str): Title of the expander.
    """
    with st.expander(expander_title):
        # Convert the dictionary into a DataFrame for display
        metadata_df = pd.DataFrame(metadata.items(), columns=["Information", "Value"])
        metadata_df = metadata_df.set_index("Information")
        
        # Display the table
        st.table(metadata_df)
