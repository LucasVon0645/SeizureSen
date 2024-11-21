import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

#TODO: Load all the files


#Code for a singular file. I will extend it for all files
interictal_data = scipy.io.loadmat("Dog_1_interictal_segment_0001.mat")
print("Keys in Interictal data:")
print(interictal_data.keys())

#Extracting the interictal segment which is the only useful part
interictal_segment = interictal_data['interictal_segment_1']


print("Interictal Segment Shape:", interictal_segment.shape)

print("Interictal Segment Data Type:", interictal_segment.dtype)

#since the shape of interical_segment is (1,1), the data is wrapped in a nested structure. It contains the following entities:
#data: This is the actual EEG segment in a 2D array
#data_length_sec: duration of each segment in seconds
#sampling_frequency: EEG signal's sampling frequency
#channels: Names of 16 channels (channel -> electrode use in EEG recording)
#sequence: numeric identifier of the sequence

#I am using this to unpack the single element from the wrapped data
segment = interictal_segment[0,0]

#This is a matrix of EEG data with shape (16, 239766).
#16 is the number of channels and 239766 is the total samples in each channel
eeg_data=segment['data']
print(eeg_data)
print("EEG Data Shape:", eeg_data.shape)

data_length = segment['data_length_sec'][0,0]
print(f"Data length: {data_length}s")

sampling_frequency = segment['sampling_frequency'][0,0]
print(f"Sampling frequency: {sampling_frequency} Hz")

#Channel names are in an array of shape (1,16) which I converted into a list.
channels = segment['channels']
channels = channels.flatten().tolist()
print(channels[0])

sequence = segment['sequence'][0,0]
print(f"Sequence: {sequence}")

#Uncomment the following if you want to visualize EEG segments for all 16 channels
"""
for i in range(eeg_data.shape[0]):
    plt.plot(eeg_data[i,:], label=channels[i])
    plt.title(f"EEG Signal (Channel {i+1})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
"""

