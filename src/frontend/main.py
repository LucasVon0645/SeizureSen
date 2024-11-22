import streamlit as st
from scipy.io import loadmat

st.title("SeizureSen")
st.write("Select an iEEG Signal File")

uploaded_file = st.file_uploader(label="Select an iEEG Signal File to be analyzed", type=["mat"])

if uploaded_file is not None:
    try:
        segment_mat = loadmat(uploaded_file)
        # Dog_1_test_segment_0001.mat
        segment = segment_mat['test_segment_1'][0,0]
        
        print(segment)
        
        eeg_data=segment['data']

        data_length = segment['data_length_sec'][0,0]
        st.write(f"Segment time length: {data_length}s")

        sampling_frequency = segment['sampling_frequency'][0,0]
        st.write(f"Sampling frequency: {sampling_frequency:.2f} Hz")
        
        channels = segment['channels']
        channels = channels.flatten().tolist()
        st.write(channels)

        # sequence = segment['sequence'][0,0]
        # print(f"Sequence: {sequence}")
    except Exception as e:
        st.error(f"An error occurred {e}")
