import os
import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.FrontEnd.utils.utils import (
    display_eeg_classes,
    display_metadata_in_expander,
    load_and_prefilter_eeg,
    load_css,
    simulate_streaming
)
from src.SeizureSenPredictor.SeizureSenPredictor import SeizureSenPredictor

def app(seizure_sen_predictor: SeizureSenPredictor):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))

    dog_image_path = os.path.join(script_dir, "images/logo.png")

    # Inject custom CSS from the external file
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

    # Heading and left-aligned image
    col1, col2 = st.columns(
        [0.2, 2]
    )  # Creating two columns, one for the heading, the other for the image.

    # Adding heading in the second column.
    col2.markdown(
        """
    <h1 class="main-title" style="font-size: 1.5em;">
        Seizure Prediction App
    </h1>
    """,
        unsafe_allow_html=True,
    )

    # Loading the image in the first column
    try:
        # Opening the image
        dog_image = Image.open(dog_image_path)
        col1.image(dog_image, width=80)
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
            sampling_freq = 400
            eeg_data, metadata, labels = load_and_prefilter_eeg(uploaded_file, sampling_freq)
            time_duration_per_label = 600  # 10-minute intervals

            display_metadata_in_expander(metadata)
            if labels is not None:
                display_eeg_classes(labels, time_per_label=time_duration_per_label)

            sampling_freq = metadata["sampling_frequency"]
            
            chunk_period = 5  # 5-second chunks
            chunk_size = sampling_freq * chunk_period  # number of samples in each chunk
            
            time_steps_model = seizure_sen_predictor.get_model_time_steps()
            segment_pred_duration = chunk_period * time_steps_model
            
            total_number_of_pred_per_label = time_duration_per_label//segment_pred_duration
            labels = np.repeat(labels, total_number_of_pred_per_label)
            
            total_chunks = int(
                        eeg_data.shape[1] / chunk_size
                    )  # Calculate total number of chunks
            
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
            elif len(selected_channels) > 3:
                st.sidebar.error("You can select up to 3 channels only.")
            else:
                update_period = st.sidebar.slider(
                    "Update Period (in seconds)", 0.1, 1.0, 0.2
                )
                window_duration = st.sidebar.slider(
                    "Window Duration (simulation seconds)", 10, 100, 60
                )
                window_size = int(window_duration * sampling_freq)
            
                st.sidebar.write(
                    f"Time slice: {chunk_period:.2f} simulation seconds"
                )
                
                simulation_duration = total_chunks * update_period
                st.sidebar.write(
                    f"Simulation duration in real world: {simulation_duration:.2f} seconds"
                )

                placeholder = st.empty()
                prediction_placeholder = st.empty()
                pred_history = []   # List to store prediction history
                segment_pred_count = 0
                
                if st.sidebar.button("Generate", key="Generate"):
                    # Initialize an empty array to store cumulative data
                    cumulative_data = np.empty((eeg_data.shape[0], 0))
                    chunk_index = (
                        0  # index used to allow the visualization of the progress bar
                    )
                    progress_bar = st.sidebar.progress(
                        0, text="Simulation Progress"
                    )  # Initialize progress bar
                    
                    # Initialize time chunks list and prediction history
                    time_chunks_list = [] # List to store time chunks that will be used at the same time to make a prediction
                                        
                    for chunk in simulate_streaming(
                        eeg_data, chunk_size, update_period
                    ):
                        time_chunks_list.append(chunk)
                        
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
                        n_rows = 1
                        n_cols = 3
                        
                        # Create subplots 
                        fig = make_subplots(
                            rows=n_rows,
                            cols=n_cols,
                            vertical_spacing=1.0,  
                            horizontal_spacing=0.1,
                            subplot_titles=[f"Channel {metadata['channels'][idx]}" for idx in selected_indices]
                        )
                        fig.update_layout(
                            height=300, 
                            margin=dict(l=5, r=5, t=10, b=5),  
                            font=dict(size=10),  
                            
                        )
                        
                        # Check if we have enough time chunks to make a prediction
                        if len(time_chunks_list) == time_steps_model:
                            true_label = labels[segment_pred_count]
                            segment_pred_count += 1
                            pred, prob = seizure_sen_predictor.classify_eeg(time_chunks_list)
                            # Calculate the interval
                            start_time = len(pred_history) * len(time_chunks_list) * chunk_period
                            end_time = (chunk_index+1) * chunk_period
                            interval = f"{int(start_time)}-{int(end_time)} s"
                            pred_history.append((interval,pred, prob, true_label))
                            with prediction_placeholder.container():
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Convert prediction history to a DataFrame for clear visualization
                                    df = pd.DataFrame(pred_history, columns=["Interval", "Prediction", "Probability", "True Label"])
                                    df["Prediction"] = df["Prediction"].map({0: "interictal", 1: "preictal"})
                                    df["True Label"] = df["True Label"].map({0: "interictal", 1: "preictal"})
                                    st.markdown("##### Classification History", unsafe_allow_html=True)
                                    st.markdown(f"<style>table{{font-size:12px;width:80%!important;margin:0 auto;}}th,td{{padding:4px 8px!important;}}</style>{df.style.hide(axis='index').to_html()}", unsafe_allow_html=True)
                                    
                                with col2:
                                    st.markdown("##### Warning System", unsafe_allow_html=True)
                                    last_pred = df.iloc[-1]["Prediction"]
                                    interval_seconds = int(end_time - start_time)
                                    prob = df.iloc[-1]["Probability"]
                                    st.markdown(f"Classification of last {interval_seconds}s: {last_pred} with probability {prob:.3f}", unsafe_allow_html=True)
                                    if pred == 1:
                                        st.error("A seizure will likely happen!", icon="🚨")
                                    else:
                                        st.warning("No seizure predicted in the moment")
                            
                            time_chunks_list = [] # Reset time chunks list

                        with placeholder.container():
                            # Update progress bar
                            progress_bar.progress(
                                chunk_index / total_chunks, text="Simulation Progress"
                            )
                            rows_cols_index_grid = [(1, 1), (1, 2), (1, 3)]

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
                                # Customize x-axis for each subplot
                                fig.update_xaxes(
                                    title_text="Time (s)",
                                    row=row,
                                    col=col,
                                    title_standoff=30,
                                )

                            
                            fig.update_layout(
                                height=300,  # Adjust the height as needed
                                plot_bgcolor='#041216',  # Set the background of the plot area
                                paper_bgcolor='#041216',  # Set the background of the entire figure
                                font=dict(color='white'),  # Ensures text remains visible
                                title=dict(
                                    text="Real-Time iEEG Signals",
                                    x=0.0,  # Center title
                                    y=1.0,  # Move the main title up
                                    xanchor='left',
                                    yanchor='top',
                                    font=dict(size=16, color='white')  # Adjust font size and color
                                ),
                                margin=dict(l=10, r=10, t=30, b=10),  # Reduce outer margins
                                showlegend=False,  # Disable legend to avoid clutter
                                autosize=True
                            )
                            
                            fig.update_annotations(yshift=-10)

                            st.plotly_chart(
                                fig, use_container_width=True, key=f"plot_{chunk_index}"
                            )

                            chunk_index += 1  # Update chunk index for visualization bar

                    col2.success("Simulation completed!")
                        

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.sidebar.info("Awaiting file upload...")
