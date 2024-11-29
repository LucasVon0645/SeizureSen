import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

from src.FrontEnd.utils.utils import (
    display_metadata_in_expander,
    extract_eeg_from_specific_file,
)


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


def app():
    """
    Frontend web application for seizure prediction in dogs.

    This function defines the interactive Streamlit-based web application where
    users can explore and test the behavior of a classification model designed
    to predict seizures in dogs. Users can upload EEG data, visualize real-time
    signal streams, and observe model predictions in response to the data.

    Features:
        - Upload EEG data for testing.
        - Simulated real-time streaming of EEG signals.
        - Visualize signals and predictions dynamically.
        - Interactive widgets to adjust streaming parameters (e.g., chunk size, update rate).

    Note:
        Ensure that all necessary dependencies (e.g., Streamlit, Plotly, NumPy) are
        installed and that the backend model is properly configured for predictions
        (data and pretrained model files are available).
    """
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
            data = extract_eeg_from_specific_file(uploaded_file)
            eeg_data = data["eeg_data"]
            metadata = data["metadata"]

            display_metadata_in_expander(metadata)

            simulation_speed = st.sidebar.slider("Simulation Speed", 10, 100, 50)

            update_period = st.sidebar.slider(
                "Update Period (in seconds)", 0.1, 1.0, 0.5
            )

            sampling_freq = data["metadata"]["sampling_frequency"]
            total_duration = data["metadata"]["duration"]

            chunk_size = round(sampling_freq * update_period * simulation_speed)

            chunk_period = chunk_size / sampling_freq

            selected_channels = st.sidebar.multiselect(
                "Select Channels to Plot",
                options=metadata["channels"],
            )
            selected_indices = [
                metadata["channels"].index(ch) for ch in selected_channels
            ]

            # Validate channel selection
            if len(selected_channels) == 0:
                st.sidebar.error("Please select at least one channel.")
            elif len(selected_channels) > 4:
                st.sidebar.error("You can select up to 4 channels only.")
            else:
                simulation_duration = total_duration / simulation_speed

                window_duration = st.sidebar.slider(
                    "Window Duration (seconds)", 10, 100, 60
                )
                window_size = int(window_duration * sampling_freq)

                st.sidebar.write(
                    f"Simulation duration: {simulation_duration:.2f} seconds"
                )
                st.sidebar.write(
                    f"New added segment duration: {chunk_period:.2f} seconds"
                )

                placeholder = st.empty()

                if st.sidebar.button("Generate", key="Generate"):
                    # Initialize an empty array to store cumulative data
                    cumulative_data = np.empty((eeg_data.shape[0], 0))
                    total_chunks = int(
                        eeg_data.shape[1] / chunk_size
                    )  # Calculate total number of chunk
                    chunk_index = (
                        0  # index used to allow the visualization of the progress bar
                    )
                    progress_bar = st.progress(
                        0, text="Simulation Progress"
                    )  # Initialize progress bar

                    for chunk in simulate_streaming(
                        eeg_data, chunk_size, update_period
                    ):
                        # Concatenate new chunk to cumulative data along the second axis
                        cumulative_data = np.concatenate(
                            (cumulative_data, chunk), axis=1
                        )
                        cumulative_time_values = (
                            np.arange(cumulative_data.shape[1]) / sampling_freq
                        )

                        data_to_plot = cumulative_data[:, -window_size:]
                        time_values_to_plot = cumulative_time_values[-window_size:]

                        # Grid for eeg signals
                        n_rows = 2
                        n_cols = 2

                        with placeholder.container():
                            # Update progress bar
                            progress_bar.progress(
                                chunk_index / total_chunks, text="Simulation Progress"
                            )

                            # Create subplots with shared x-axis
                            fig = make_subplots(
                                rows=n_rows,
                                cols=n_cols,
                                vertical_spacing=0.2,  # Adjust spacing between subplots
                                horizontal_spacing=0.1,
                                subplot_titles=[
                                    f"Channel {metadata['channels'][idx]}"
                                    for idx in selected_indices
                                ],
                            )

                            rows_cols_index_grid = [(1, 1), (1, 2), (2, 1), (2, 2)]

                            for idx, channel_index in enumerate(selected_indices):
                                row, col = rows_cols_index_grid[idx]

                                fig.add_trace(
                                    go.Scatter(
                                        x=time_values_to_plot,
                                        y=data_to_plot[channel_index],
                                        mode="lines",
                                        name=f"Channel {metadata['channels'][channel_index]}",
                                    ),
                                    row=row,
                                    col=col,
                                )
                                # Customize y-axis for each subplot
                                fig.update_yaxes(
                                    title_text="Amplitude (µV)", row=row, col=col
                                )
                                fig.update_xaxes(
                                    title_text="Time (s)",
                                    row=row,
                                    col=col,
                                    title_standoff=15,
                                )

                            # Update layout
                            fig.update_layout(
                                title="Real-Time EEG Signals",
                                showlegend=False,  # Disable legend to avoid clutter,
                                height=500,
                            )

                            st.plotly_chart(
                                fig, use_container_width=True, key=f"plot_{chunk_index}"
                            )

                            chunk_index += 1  # Update chunk index for visualization bar

                    st.success("Simulation completed!")

        except Exception as e:
            st.error(f"Error loading the file: {e}")

    else:
        st.sidebar.info("Awaiting file upload...")
