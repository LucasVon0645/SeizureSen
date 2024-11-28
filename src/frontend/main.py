import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import os
import time

import src.DataExtraction.DataExtractor as data_extractor

from src.FrontEnd.utils.utils import display_metadata_in_expander


def simulate_streaming(data, chunk_size, sleep_time):
    # chunk = data[:][0:5000]
    # print(chunk.shape)
    # yield chunk
    # For now, we are commenting on this to avoid processing delays.

    for i in range(0, data.shape[1], chunk_size):
        chunk = data[:, i:i + chunk_size]
        yield chunk
        time.sleep(sleep_time)

# Main function
def main():
    st.set_page_config(
        page_title="EEG Seizure Prediction", page_icon=":dog:", layout="wide"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))

    dog_image_path = os.path.join(script_dir, "images/logo.png")

    # Customizing the style to match the blue and black theme.
    image_style = """
    <style>
        .dog-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
            border-radius: 15px;
        }
        body {
            background-color: #ffffff;  # Black background
            color: #ffffff;  # White text
        }
    </style>
"""
    # Inserting the image with the custom styling.
    st.markdown(image_style, unsafe_allow_html=True)

    # Heading and left-aligned image
    col1, col2 = st.columns(
        [0.2, 2]
    )  # Creating two columns, one for the heading, the other for the image.

    # Adding heading in the second column.
    col2.markdown(
        """
    <h1 style='
        font-size: 2.6em; 
        color: #999999;
        font-family: "Comic Sans MS"; 
        font-weight: bold'>
        Seizure Prediction App
    </h1>
    """,
        unsafe_allow_html=True,
    )

    # Loading the image in the first column
    try:
        # Opening the image
        dog_image = Image.open(dog_image_path)
        col1.image(dog_image, width=110)
    except Exception as e:
        # We can comment this as the image is  optional(for aesthetic reasons).
        st.error(f"Error loading the image: {e}")
        st.title("EEG Signal Viewer and Preictal Detection")

    # Uploading the file.
    st.sidebar.header("Upload EEG File")
    # Specifying the type of file.
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["mat"])

    if uploaded_file:
        st.sidebar.success("File uploaded successfully!")
        try:
            data = data_extractor.extract_segment_from_specific_file(uploaded_file)
            eeg_data = data["eeg_data"]
            metadata = data["metadata"]

            st.success("EEG Signal Loaded Successfully!")

            display_metadata_in_expander(metadata)
            
            simulation_speed = st.sidebar.slider(
                "Simulation Speed", 10, 100, 50
            )
            
            sleep_time = st.sidebar.slider(
                "Time interval per chunk (in seconds)", 0.1, 2.0, 0.5
            )
            
            sampling_freq = data["metadata"]["sampling_frequency"]
            total_duration = data["metadata"]["duration"]
            
            chunk_size = round(sampling_freq * sleep_time * simulation_speed)
            
            channel_option = st.sidebar.selectbox("Select a Channel", options=metadata["channels"])
            channel_index = metadata["channels"].index(channel_option)
            
            simulation_duration = total_duration / simulation_speed
            
            st.sidebar.write(f"Simulation duration: {simulation_duration:.2f} seconds")

            placeholder = st.empty()

            if st.sidebar.button("Generate", key="Generate"):
                # Initialize an empty array to store cumulative data
                cumulative_data = np.empty((eeg_data.shape[0], 0))
                
                for chunk in simulate_streaming(eeg_data, chunk_size, sleep_time):
                    # Concatenate new chunk to cumulative data along the second axis
                    cumulative_data = np.concatenate((cumulative_data, chunk), axis=1)
                    time_values = np.arange(cumulative_data.shape[1]) / sampling_freq
                    
                    with placeholder.container():
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(x=time_values, y=cumulative_data[channel_index], mode="lines", name="EEG Signal")
                        )

                        fig.update_layout(
                            title=f"Real-Time EEG Signal Channel {channel_option}",
                            xaxis_title="Time (s)",
                            yaxis_title="Amplitude (µV)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.write("Simulation completed!")

        except Exception as e:
            st.error(f"Error loading the file: {e}")

    else:
        st.sidebar.info("Awaiting file upload...")


if __name__ == "__main__":
    main()
