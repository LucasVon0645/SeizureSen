import os

import numpy as np
import scipy.signal

import tensorflow as tf


def eeg_slices(eeg_data, sampling_freq, window_duration):
    """
    Slices EEG data into smaller segments based on the specified window duration.
    Parameters:
    eeg_data (numpy.ndarray): The EEG data to be sliced, with shape (channels, samples).
    sampling_freq (int): The sampling frequency of the EEG data in Hz.
    window_duration (float): The duration of each window in seconds.
    Returns:
    list: A list of numpy.ndarray, each containing a slice of the EEG data.
    """

    slice_samples = int(sampling_freq * window_duration)  # no. of samples per slice
    total_samples = eeg_data.shape[1]  # total no. of samples
    slices = []
    for i in range(0, total_samples, slice_samples):
        if (
            i + slice_samples <= total_samples
        ):  # skipping the last slice if it doesn't fit in 30s window
            slices.append(eeg_data[:, i : i + slice_samples])
    return slices


def overlap(x, y, only_class_1=False):
    """
    Used to enhance imbalanced data by combining the first half and the second half of different samples
    to generate new samples.

    Parameters:
        x: np.ndarray
            Input feature data, shape is [number of samples, number of channels, number of timesteps].
        y: np.ndarray
            Corresponding labels, shape is [number of samples].

    Returns:
        x_augmented: np.ndarray
            Enhanced feature data.
        y_augmented: np.ndarray
            Enhanced labels.
    """
    shape_x = x.shape
    zero_indices = np.where(y == 0)[0].tolist()  # Find the indices of samples with label 0
    one_indices = np.where(y == 1)[0].tolist()  # Find the indices of samples with label 1

    # Divide the timesteps into two halves
    time_steps = shape_x[2]  # The third dimension represents the number of timesteps
    first_part = time_steps // 2

    # Initialize augmented data
    tmp_x = []
    tmp_y = []

    if not only_class_1:
        # Augment samples with label 0
        for i in range(len(zero_indices) - 1):
            tmp = np.concatenate(
                (x[zero_indices[i], :, :first_part], x[zero_indices[i + 1], :, first_part:]),
                axis=1
            )
            tmp_x.append(tmp)
            tmp_y.append(0)

    # Augment samples with label 1
    for i in range(len(one_indices) - 1):
        tmp = np.concatenate(
            (x[one_indices[i], :, :first_part], x[one_indices[i + 1], :, first_part:]),
            axis=1
        )
        tmp_x.append(tmp)
        tmp_y.append(1)

    # Convert the augmented data to numpy arrays
    tmp_x = np.array(tmp_x)
    tmp_y = np.array(tmp_y)

    # Combine the original data with the augmented data
    x_augmented = np.concatenate((x, tmp_x), axis=0)
    y_augmented = np.concatenate((y, tmp_y), axis=0)

    return x_augmented, y_augmented


def save_preprocessed_data(directory, filename="preprocessed_data.npz", **data):
    """
    Save preprocessed data to a specified directory in .npz format.
    Parameters:
    directory (str): The directory where the file will be saved.
    filename (str, optional): The name of the file to save the data. Defaults to "preprocessed_data.npz".
    **data: Arbitrary keyword arguments representing the data to be saved.
    Raises:
    ValueError: If no data is provided to save.
    Example:
    save_preprocessed_data('/path/to/save', data1=array1, data2=array2)
    """

    if not data:
        raise ValueError("No data provided to save.")

    # Ensure the directory exists
    save_dir = os.path.normpath(directory)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, filename)

    # Save data
    np.savez(file_path, **data)
    print(f"Data saved to {file_path}")


def load_preprocessed_data(data_directory, file_name="preprocessed_data.npz"):
    """
    Load preprocessed data from a specified directory and file.
    Parameters:
    data_directory (str): The directory where the preprocessed data file is located.
    file_name (str, optional): The name of the preprocessed data file. Defaults to "preprocessed_data.npz".
    Returns:
    dict: A dictionary containing the loaded data.
    Raises:
    FileNotFoundError: If the specified file does not exist.
    Example:
    data = load_preprocessed_data("data")
    print(data.keys())
    dict_keys(['key1', 'key2', 'key3'])
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.normpath(os.path.join(script_dir, "..", "..", data_directory))
    file_path = os.path.join(load_dir, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data
    data = np.load(file_path, allow_pickle=True)
    print(f"Data loaded from {file_path}")
    return {key: data[key] for key in data}


def get_frequency_bands():
    """
    Returns a dictionary of EEG frequency bands with their corresponding frequency ranges in Hz.
    Frequency bands:
        - delta: (0.1, 4) - Deep sleep, restorative sleep, and unconscious brain activity.
        - theta: (4, 8) - Relaxation, light sleep, drowsiness, and meditation, often linked to creativity.
        - alpha: (8, 12) - Calm wakefulness, relaxed state, and idle mental processes, associated with resting but alert.
        - beta: (12, 30) - Active thinking, problem-solving, decision-making, and focused mental activity.
        - low_gamma: (30, 50) - Higher cognitive functions, learning, and memory processing.
        - mid_gamma: (50, 70) - Advanced problem-solving, information processing, and heightened perception.
        - high_gamma_1: (70, 100) - Enhanced brain activity during tasks requiring attention or conscious focus.
        - high_gamma_2: (100, 180) - Intense brain activity during high-level processing and cognitive tasks.
    Returns:
        dict: A dictionary where keys are frequency band names and values are tuples representing the frequency range in Hz.
    """

    return {
        "delta": (0.1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "low_gamma": (30, 50),
        "mid_gamma": (50, 70),
        "high_gamma_1": (70, 100),
        "high_gamma_2": (100, 180),
    }


def divide_into_frequency_chunks(eeg_data, sampling_freq):
    """
    Divide EEG data into frequency chunks based on predefined frequency bands.
    Parameters:
    eeg_data (numpy.ndarray): The EEG data array of shape (n_channels, n_samples).
    sampling_freq (float): The sampling frequency of the EEG data in Hz.
    Returns:
    dict: A dictionary where keys are frequency band names and values are the FFT magnitudes
          within those frequency bands for each channel.
    """

    frequency_bands = get_frequency_bands()
    num_samples = eeg_data.shape[1]
    freqs = np.fft.rfftfreq(num_samples, d=1 / sampling_freq)  # Frequency bins
    fft_magnitude = np.abs(np.fft.rfft(eeg_data, axis=1))  # FFT Magnitude

    frequency_band_data = {}

    for band_name, (low, high) in frequency_bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_data = fft_magnitude[:, band_mask]  # FFT magnitudes in the frequency band
        frequency_band_data[band_name] = band_data

    return frequency_band_data


def calculate_band_features(frequency_band_data, use_std_in_time_domain=False, eeg_slice=None):
    """
    Calculate band features for each frequency band.
    Parameters:
    frequency_band_data (dict): A dictionary containing EEG data in different frequency bands.
    use_std_in_time_domain (bool, optional): Whether to include the standard deviation of the EEG slice in the time domain.
    eeg_slice (numpy.ndarray, optional): The EEG slice to calculate the standard deviation in the time domain.
    Returns:
    dict: A dictionary containing band features for each band. The keys are in the format "<band_name>_<feature_name>".
          The values are arrays containing the feature values for each channel.
    """
    
    band_features = (
        {}
    )  # Band features are the mean log amplitude and standard deviation.
    # This function returns a dictionary containing band features for each band

    for band_name, band_data in frequency_band_data.items():
        log_amplitude = np.log10(np.clip(band_data, a_min=1e-10, a_max=None))

        mean_log_amplitude = np.mean(log_amplitude, axis=1)  # Mean for each channel

        band_features[f"{band_name}_mean"] = mean_log_amplitude
        if not use_std_in_time_domain:
            std_log_amplitude = np.std(
                log_amplitude, axis=1
            )  # Standard deviation for each channel
            band_features[f"{band_name}_std"] = std_log_amplitude

    if use_std_in_time_domain:
        if eeg_slice is None:
            raise ValueError(
                "eeg_slice must be provided to calculate the standard deviation in the time domain."
                )
        
        # Calculate the standard deviation of the EEG slice in the time domain
        # for each channel and add it to the band features
        std_time_domain = np.std(eeg_slice, axis=1)
        band_features["time_domain_std"] = std_time_domain

    return band_features


def resample(eeg_segment, new_sampling_freq, time_length=600):
    """
    Resamples the given EEG data to a new sampling rate.

    Parameters:
    eeg_data (numpy.ndarray): The EEG data to be resampled. Expected to be a 2D array where rows represent channels and columns represent time points.
    newSamplingRate (int): The new sampling rate in Hz.
    timeLength (int, optional): The length of time in seconds for which the data is to be resampled. Default is 600 seconds.

    Returns:
    numpy.ndarray: The resampled EEG data.
    """
    return scipy.signal.resample(eeg_segment, new_sampling_freq * time_length, axis=1)


def butterworth_bandpass_filter(eeg_segment=np.ndarray, order=int, band=None, sampling_freq=400):
    """
    Applies a Butterworth bandpass filter to an EEG signal segment.

    Parameters:
    - eeg_segment: np.ndarray
        The input EEG signal, typically a 2D array (channels x samples).
    - order: int
        The order of the Butterworth filter (higher values = sharper cutoff).
    - band: list of float, optional (default: [0.1, 180])
        The passband frequencies [low_cutoff, high_cutoff] in Hz.
    - frequency: float, optional (default: 400)
        The sampling frequency of the EEG data in Hz.

    Returns:
    - np.ndarray
        The filtered EEG signal.
    """

    if band is None:
        band = [0.1, 180]
    # Compute the filter coefficients for the Butterworth bandpass filter
    b, a = scipy.signal.butter(
        order, np.array(band) / (sampling_freq / 2.0), btype="band"
    )

    # Apply the filter to the EEG data along the time axis (axis=1)
    return scipy.signal.lfilter(b, a, eeg_segment, axis=1)


def resample_and_apply_bandpass_filter(eeg_segment, order=5, band=None, new_sampling_freq=400, time_length=600) -> tuple[np.ndarray, int]:
    """
    Resamples the given EEG data to a new sampling rate and applies a Butterworth bandpass filter.

    Parameters:
    - eeg_segment: dict
        A dictionary containing the EEG data and its label. Expected keys are "eeg_data"
        (2D array where rows represent channels and columns represent time points) and "label".
    - order: int (default: 5)
        The order of the Butterworth filter (higher values = sharper cutoff).
    - band: list of float, optional (default: [0.1, 180])
        The passband frequencies [low_cutoff, high_cutoff] in Hz.
    - new_sampling_freq: int
        The new sampling rate in Hz.
    - time_length: int, optional (default: 600)
        The length of time in seconds for which the data is to be resampled.

    Returns:
    - tuple
        A tuple containing the resampled and filtered EEG data (np.ndarray) and the label.
    """
    eeg_data = eeg_segment["eeg_data"]
    label = eeg_segment["label"]

    resampled_data = resample(eeg_data, new_sampling_freq, time_length)
    filtered_data = butterworth_bandpass_filter(
        resampled_data, order, band, new_sampling_freq
    )

    return filtered_data, label


def transform_to_tensor(features: list[dict], labels: list[int], steps: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Transform a list of dictionaries into a 4D tensor and group labels.

    Parameters:
        features: list of dictionaries
            Each dictionary contains keys (features) mapping to arrays (channels).
        steps: int
            Number of slices to aggregate into one sample slot.
        labels: list of int
            List of labels corresponding to each slice. Must be the same length as data.

    Returns:
        tuple: (4D keras tensor, 1D tensor of labels)
            Shape of tensor: (n_samples, n_channels, features, steps)
            Shape of labels_tensor: (n_samples,)

    Raises:
        ValueError: If the total number of slices is not divisible by the number of
        steps or if labels within a group are not the same.
    """
    if len(features) != len(labels):
        raise ValueError("The length of features and labels must be the same.")

    # Extract keys (features) and validate consistency
    feature_keys = features[0].keys()
    n_channels = len(features[0][list(feature_keys)[0]])  # Number of channels

    # Convert the list of dict into a list of tensors. All features are put together in a tensor
    all_slices = []
    for slice_dict in features:
        slice_features = tf.convert_to_tensor(
            [slice_dict[key] for key in feature_keys], dtype=tf.float32
        )  # Shape: (features, n_channels)
        all_slices.append(
            tf.transpose(slice_features)
        )  # Transpose to (n_channels, features)

    # Stack all slices into a 3D tensor (total_slices, n_channels, features)
    all_slices = tf.stack(all_slices)  # Shape: (total_slices, n_channels, features)

    # Ensure the total number of slices is divisible by steps
    total_slices = all_slices.shape[0]
    n_samples = total_slices // steps  # Number of sample slots

    # Ensure the total number of slices is divisible by steps
    excess_slices = total_slices % steps
    if excess_slices != 0:
        print(f"Warning: Discarding {excess_slices} excess slices.")
        all_slices = all_slices[:-excess_slices]
        labels = labels[:-excess_slices]

    # Reshape into (n_samples, steps, n_channels, features)
    reshaped_slices = tf.reshape(all_slices, (n_samples, steps, n_channels, -1))

    # Transpose to (n_samples, n_channels, features, steps)
    tensor = tf.transpose(reshaped_slices, perm=[0, 2, 3, 1])

    # Group labels
    labels_tensor = []
    for i in range(n_samples):
        group_labels = labels[i * steps : (i + 1) * steps]
        if len(set(group_labels)) != 1:
            raise ValueError("All slices in a group must have the same label.")
        labels_tensor.append(group_labels[0])

    labels_tensor = tf.convert_to_tensor(labels_tensor, dtype=tf.int32)

    return tensor, labels_tensor

def transform_features_to_tensor(features: list[dict], steps: int) -> tf.Tensor:
    """
    Transform a list of dictionaries into a 4D tensor and group labels.

    Parameters:
        features: list of dictionaries
            Each dictionary contains keys (features) mapping to arrays (channels).
        steps: int
            Number of slices to aggregate into one sample slot.

    Returns:
        tensor: 4D keras tensor (n_samples, n_channels, features, steps)

    Raises:
        ValueError: If the total number of slices is not divisible by the number of
        steps or if labels within a group are not the same.
    """

    # Extract keys (features)
    feature_keys = features[0].keys()
    n_channels = len(features[0][list(feature_keys)[0]])  # Number of channels

    # Convert the list of dict into a list of tensors. All features are put together in a tensor
    all_slices = []
    for slice_dict in features:
        slice_features = tf.convert_to_tensor(
            [slice_dict[key] for key in feature_keys], dtype=tf.float32
        )  # Shape: (features, n_channels)
        all_slices.append(
            tf.transpose(slice_features)
        )  # Transpose to (n_channels, features)

    # Stack all slices into a 3D tensor (total_slices, n_channels, features)
    all_slices = tf.stack(all_slices)  # Shape: (total_slices, n_channels, features)

    # Ensure the total number of slices is divisible by steps
    total_slices = all_slices.shape[0]
    if total_slices % steps != 0:
        raise ValueError(
            "The total number of slices must be divisible by the number of steps."
        )

    # Stack all slices into a 3D tensor (total_slices, n_channels, features)
    all_slices = tf.stack(all_slices)  # Shape: (total_slices, n_channels, features)

    # Ensure the total number of slices is divisible by steps
    total_slices = all_slices.shape[0]
    if total_slices % steps != 0:
        raise ValueError(
            "The total number of slices must be divisible by the number of steps."
        )

    n_samples = total_slices // steps  # Number of sample slots

    # Reshape into (n_samples, steps, n_channels, features)
    reshaped_slices = tf.reshape(all_slices, (n_samples, steps, n_channels, -1))

    # Transpose to (n_samples, n_channels, features, steps)
    tensor = tf.transpose(reshaped_slices, perm=[0, 2, 3, 1])

    return tensor

def scale_across_time_tf(data: tf.Tensor, scalers: list[dict]) -> tf.Tensor:
    """
    Scales data across time using precomputed scalers.
    The features in each channel are scaled independently using the mean and standard deviation computed for that channel and feature.
    Parameters:
        - data: Input data tensor. Shape (samples, channels, bins, time_steps).
        - scalers: A list of dictionaries containing keys 'mean' and 'std'. Each dictionary corresponds to a channel.
                   Both 'mean' and 'std' are 1D arrays with the same length as the number of features.
    Returns:
        - scaled_data: The scaled data as a TensorFlow tensor.
    """
    sample_num, channel_num, bin_num, time_step_num = data.shape

    # Validate scalers
    if len(scalers) != channel_num:
        raise ValueError("The number of scalers must match the number of channels.")

    # Scale each channel independently
    scaled_data = tf.TensorArray(tf.float32, size=channel_num)

    for i in range(channel_num):
        # Extract scalers for this channel
        mean = scalers[i]['mean']
        std = scalers[i]['std']

        # Reshape and scale the data for this channel
        channel_data = tf.reshape(
            tf.transpose(data[:, i, :, :], perm=[0, 2, 1]),
            [sample_num * time_step_num, bin_num]
        )
        scaled_channel_data = (channel_data - mean) / std
        scaled_channel_data = tf.transpose(
            tf.reshape(scaled_channel_data, [sample_num, time_step_num, bin_num]),
            perm=[0, 2, 1]
        )

        # Add scaled data back to TensorArray
        scaled_data = scaled_data.write(i, scaled_channel_data)

    # Stack all channels back into the tensor
    # The original dimension for channel is axis=1, so we stack along axis=1
    scaled_data = tf.stack(scaled_data.stack(), axis=1)

    return scaled_data

def augment_raw_train_slices(train_slices_with_label: list[tuple[np.ndarray, int]], only_class_1 = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment the raw training slices using the overlap technique.
    Parameters:
        train_slices_with_label: list of tuples
            A list of tuples containing the raw training slices and their corresponding labels.
    Returns:
        train_slices_with_label_augmented: list of tuples
            A list of tuples containing the augmented training slices and their corresponding labels.
    """
    
    # Data Augmentation: Call the overlap function
    print("Applying data augmentation in training data using overlap...")
    if only_class_1:
        print("Augmenting only class 1 samples...")

    train_slices, train_labels = zip(*train_slices_with_label)  # Unpack slices and labels
    train_slices = np.array(train_slices)
    train_labels = np.array(train_labels)
    
    # Call the overlap function
    train_slices_augmented, train_labels_augmented = overlap(train_slices, train_labels, only_class_1)

    # Repack the augmented data
    train_slices_with_label_augmented = list(zip(train_slices_augmented, train_labels_augmented))
    
    print(f"Original training samples: {len(train_slices_with_label)}")
    print(f"Augmented training samples: {len(train_slices_with_label_augmented)}")
    
    return train_slices_with_label_augmented