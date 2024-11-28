import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


class DataExtractor:
    """
    This DataExtractor class will load and manage EEG segments
    """

    def __init__(self, data_directory):
        """
        Initialize the DataExtractor class with root directory
        :param data_directory: Data folder path
        """

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = os.path.normpath(
            os.path.join(script_dir, "..", "..", data_directory)
        )
        self.data = {"interictal": [], "preictal": [], "test": []}
        self.metadata = {"sampling_frequency": None, "channels": None}

    def _load_segment(self, file_path):
        """
        Load a single .mat file, extract its EEG and metadata.
        :param file_path: Path to .mat file
        :return: A dictionary containing EEG data and associated metadata.
        """
        mat_data = scipy.io.loadmat(file_path)
        key = next(key for key in mat_data.keys() if "segment" in key)
        segment = mat_data[key][0, 0]

        # Extracting data and metadata from the wrapped nested structure

        eeg_data = segment["data"]
        data_length = segment["data_length_sec"][0, 0]
        sampling_frequency = segment["sampling_frequency"][0, 0]
        channels = segment["channels"]
        channels = channels.flatten().tolist()

        return {
            "eeg_data": eeg_data,
            "data_length": data_length,
            "sampling_frequency": sampling_frequency,
            "channels": channels,
        }

    def load_data(self, dog_ids, segment_types=None):
        """
        Load specified segment types (e.g., interictal, preictal, test) for the given dog IDs
        :param dog_ids: List of dog IDs to load (e.g., ["Dog_1", "Dog_2"]).
        :param segment_type: List of segment types to load (e.g., ["interictal", "preictal", "test"]).
                            None to load all segments
        """
        if segment_types is None:
            segment_types = ["interictal", "preictal", "test"]

        for dog_id in dog_ids:
            dog_path = os.path.join(self.data_directory, dog_id)
            if not os.path.isdir(dog_path):
                print(f"Directory not found for {dog_id}. Skipping...")
                continue
            for file_name in os.listdir(dog_path):
                file_path = os.path.join(dog_path, file_name)

                # Find out which type of segment it is
                segment_type = None
                if "interictal" in file_name and "interictal" in segment_types:
                    segment_type = "interictal"
                    label = 0
                elif "preictal" in file_name and "preictal" in segment_types:
                    segment_type = "preictal"
                    label = 1
                elif "test" in file_name and "test" in segment_types:
                    segment_type = "test"
                    label = None

                if segment_type is None:
                    continue

                # Loading segment
                segment_data = self._load_segment(file_path)
                segment_data["label"] = label

                if self.metadata["sampling_frequency"] is None:
                    self.metadata["sampling_frequency"] = segment_data[
                        "sampling_frequency"
                    ]
                    self.metadata["channels"] = segment_data["channels"]

                self.data[segment_type].append(segment_data)

        print(f"Loaded data for {len(dog_ids)} dog(s)")
        for segment_type in segment_types:
            print(
                f"  - {segment_type.capitalize()} segments: {len(self.data[segment_type])}"
            )

    def get_data(self):
        """
        Retrieve the loaded data
        :return: A dictionary with interictal, preictal and test data
        """
        return self.data

    def get_metadata(self):
        """
        Retrieve metadata
        :return: A dictionary with metadata (e.g., sampling frequency, channels).
        """
        return self.metadata

def extract_segment_from_specific_file(file):
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

# USAGE

# if __name__ == "__main__":
#     data_extractor = DataExtractor(data_directory="data")
#     data_extractor.load_data(dog_ids=["Dog_1", "Dog_2"], segment_types=["interictal", "preictal"]) <- Replace segment_types=["test"] to load the test segments
#     loaded_data = data_extractor.get_data()


#     interictal_segments = loaded_data["interictal"]
#     preictal_segments = loaded_data["preictal"]
#     for segment in interictal_segments:
#         print(f"Interictal label: {segment['label']}, Shape: {segment['eeg_data'].shape}")
#     for segment in preictal_segments:
#         print(f"Preictal label: {segment['label']}, Shape: {segment['eeg_data'].shape}")
        

#     metadata = data_extractor.get_metadata()
#     print("Metadata:", metadata)
