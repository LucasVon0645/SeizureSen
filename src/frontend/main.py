import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import plotly.graph_objects as go
import scipy.io


def simulate_streaming(data, chunk_size, sleep_time):
    chunk=data[0][0:5000]
    yield chunk
    # For now, we are commenting on this to avoid processing delays.

    # for i in range(0, len(data), chunk_size):
    #     chunk = data[i:i + chunk_size]
    #     yield chunk
    #     time.sleep(sleep_time)

# Main function
def main():
    st.set_page_config(page_title="EEG Seizure Prediction", page_icon=":dog:", layout="wide")
    
    dog_image_path = 'image.png'

    # Customizing the style to match the blue and black theme
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
    # Inserting the image with the custom styling
    st.markdown(image_style, unsafe_allow_html=True)

    # Heading and left-aligned image
    col1, col2 = st.columns([.2, 2])  # Creating two columns, one for the heading, the other for the image

    # Adding heading in the second column
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
    unsafe_allow_html=True
)
    
    # Loading the image in the first column
    try:
        # Opening the image
        dog_image = Image.open(dog_image_path)
        col1.image(dog_image, width=110)  
    except Exception as e:
        # We can comment this as the image is for optional(aesthatic reasons).
        st.error(f"Error loading the image: {e}")
        st.title("EEG Signal Viewer and Preictal Detection")

    
    
    # Uploading the file
    st.sidebar.header("Upload EEG File")
    #Specifying the type of file.
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["mat"])
    
    if uploaded_file:
        st.sidebar.success("File uploaded successfully!")
        try:
            mat_data = scipy.io.loadmat(uploaded_file)
            # Dynamically searching for the 'interictal_segment_*' key.
            interictal_segment_key = None
            for key in mat_data.keys():
                if key.startswith('interictal_segment_'):
                    interictal_segment_key = key
                    break

            if interictal_segment_key is None:
                st.error("No interictal data not found in the .mat file!")
            else:
                # Extracting the data
                interictal_segment = mat_data[interictal_segment_key]
                segment = interictal_segment[0, 0]
                eeg_data = segment['data']
                st.success(f"EEG Signal Loaded Successfully from {interictal_segment_key}!")

                # Setting up chunk processing parameters
                chunk_size = st.sidebar.slider("Chunk size (number of samples)", 10, 1000, 500)
                sleep_time = st.sidebar.slider("Time interval per chunk (in seconds)", 0.1, 2.0, 0.5)

                placeholder = st.empty()
           
            # We are currently taking the chunk size and sleep time parameters but not plotting the graph based on this data. 
            # It will be implemented in the future.
            if st.sidebar.button("Generate",key="Generate"):
                for chunk in simulate_streaming(eeg_data, chunk_size, sleep_time):
                    with placeholder.container():
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                        y=chunk,  # The EEG chunk data
                        mode='lines',  # Line plot
                        name="Current Chunk"  # Legend label
                        ))
                    
                        fig.update_layout(
                        title="Real-Time EEG Signal Chunk",
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude (µV)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("Simulation complete!")
               

        
        except Exception as e:
            st.error(f"Error loading the file: {e}")
    
    else:
        st.sidebar.info("Awaiting file upload...")

if __name__ == "__main__":
    main()
