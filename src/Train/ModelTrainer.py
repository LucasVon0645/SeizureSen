import os

import joblib
import tensorflow as tf
from keras.api.utils import to_categorical
from keras.api.models import Model

from src.Train.utils import plot_training_history
from src.Preprocessing.utils import load_preprocessed_data, transform_to_tensor
from src.Model.MultiViewConvModel import MultiViewConvModel
from src.Model.utils import load_model_from_config, load_config, load_scalers_from_config

class ModelTrainer:
    """
    A class to train and evaluate a multi-view convolutional neural network model
    for EEG data analysis.
    This class loads the configuration settings, preprocessed EEG data, and
    trains the model using the training data.
    The trained model is then evaluated using the test data, and the results
    are saved to a specified file path.
    """

    def __init__(self, cfg_path: str, data_directory: str):
        self.data_path = data_directory
        self.config_path = cfg_path

        self.config = load_config(cfg_path)

        self.model: Model|None = None

        # Freq Domain Tensorflow Tensors (n_samples, n_channels, fft_bins, steps)
        self.X_train_freq: tf.Tensor|None = None
        self.X_test_freq: tf.Tensor|None = None

        # Time Domain Tensorflow Tensors (n_samples, n_channels, pca_bins, steps)
        self.X_train_time: tf.Tensor|None = None
        self.X_test_time: tf.Tensor|None = None

        # Label Tensors (n_samples,)
        self.y_train: tf.Tensor|None = None
        self.y_test: tf.Tensor|None = None

        # Scalers for both time and frequency domain 
        # Lists of dictionaries {"mean": float, "std": float}
        # One dictionary per channel
        self.scalers_time: list[dict]|None = None
        self.scalers_freq: list[dict]|None = None

    def load_data(self, file_names_dict=dict | None):
        """
        Load and preprocess EEG data for training and evaluation.
        This function loads preprocessed EEG data from specified files,
        transforms the data into tensors, and assigns them to instance
        variables for frequency and time domain data.
        Args:
            file_names_dict (dict | None): A dictionary containing file names.
                                           If None, default file names are used:
                                             {
                                                  "freq_train": "freq_domain_train.npz",
                                                  "freq_test": "freq_domain_test.npz",
                                                  "time_train": "time_domain_train.npz",
                                                  "time_test": "time_domain_test.npz",
                                            }
        Raises:
            KeyError: If any of the required keys are missing in the file_names_dict.
        """

        print("\n\nLoading preprocessed EEG data...")
        data_dir = self.data_path

        if file_names_dict is not None:
            file_names_dict = {
                "freq_train": "freq_domain_train.npz",
                "freq_test": "freq_domain_test.npz",
                "time_train": "time_domain_train.npz",
                "time_test": "time_domain_test.npz",
            }

        # Load preprocessed slices of eeg data for train and evaluation

        data_train_freq = load_preprocessed_data(data_dir, file_names_dict["freq_train"])
        data_test_freq = load_preprocessed_data(data_dir, file_names_dict["freq_test"])

        data_train_time = load_preprocessed_data(data_dir, file_names_dict["time_train"])
        data_test_time = load_preprocessed_data(data_dir, file_names_dict["time_test"])

        steps = self.config["model_time_steps"]

        # Transform the data into tensors
        # Consecutive slices with same label are grouped together (group size = steps)

        # Frequency domain data (train and test)
        self.X_train_freq, self.y_train = transform_to_tensor(
            data_train_freq["X"], data_train_freq["y"], steps
        )
        self.X_test_freq, self.y_test = transform_to_tensor(
            data_test_freq["X"], data_test_freq["y"], steps
        )

        # Time domain data (train and test)
        self.X_train_time, _ = transform_to_tensor(
            data_train_time["X"], data_train_time["y"], steps
        )
        self.X_test_time, _ = transform_to_tensor(
            data_test_time["X"], data_test_time["y"], steps
        )
       
    def load_scalers(self):
        """
        Loads the time and frequency scalers from the configuration.
        This method initializes the `scalers_time` and `scalers_freq` attributes
        by loading the scalers defined in the configuration file.
        Returns:
            None
        """
        
        self.scalers_time, self.scalers_freq = load_scalers_from_config(self.config)

    def compute_scalers_transform(self, time_domain = True):
        """
        Compute the mean and standard deviation for each channel and standardize the data.
        Parameters:
            - time_domain: If True, compute the scalers for the time domain data.
                           If False, compute the scalers for the frequency domain data.

        Obs.: The scalers are stored in the instance variables `self.scalers_time` and 
        `self.scalers_freq` as a list of dictionaries, where each dictionary contains
        the mean and standard deviation for a channel.
        """
        if time_domain:
            data = self.X_train_time
        else:
            data = self.X_train_freq

        sample_num, channel_num, bin_num, time_step_num = data.shape

        # Prepare a list to store the scalers for each channel (mean and std)
        scalers = []

        # Create a TensorArray to store the scaled data
        scaled_data = tf.TensorArray(tf.float32, size=channel_num)

        # Loop through each channel and perform standardization
        for i in range(channel_num):
            # Extract the data for the current channel and reshape for scaling
            channel_data = tf.reshape(
                tf.transpose(data[:, i, :, :], perm=[0, 2, 1]),
                [sample_num * time_step_num, bin_num]
            )

            # Compute the mean and standard deviation for the channel
            mean = tf.reduce_mean(channel_data, axis=0)
            std = tf.math.reduce_std(channel_data, axis=0)

            # Standardize the data
            standardized_channel_data = (channel_data - mean) / std

            # Store the mean and std for this channel (optional, if you need to use them later)
            scalers.append({'mean': mean, 'std': std})

            # Reshape and transpose the standardized data back to the original dimensions
            standardized_channel_data = tf.transpose(
                tf.reshape(standardized_channel_data, [sample_num, time_step_num, bin_num]),
                perm=[0, 2, 1]
            )

            # Add the standardized data to the TensorArray
            scaled_data = scaled_data.write(i, standardized_channel_data)

        # Stack all channels back into a single tensor
        scaled_data = tf.stack(scaled_data.stack(), axis=1)

        # Update the instance variables with the standardized data and scalers
        if time_domain:
            self.X_train_time = scaled_data
            self.scalers_time = scalers
        else:
            self.X_train_freq = scaled_data
            self.scalers_freq = scalers

    def save_scalers(self):
        """
        Save scalers to a file.
        """

        model_path = self.config["model_path"]
        filename = self.config["name"] + "_scalers.pkl"

        filepath = os.path.join(model_path, filename)

        scalers = {
            "time_domain": self.scalers_time,
            "frequency_domain": self.scalers_freq
        }

        # Save the dictionary to a file using joblib
        joblib.dump(scalers, filepath)

    #! implement the augment_train_data method
    def augment_train_data(self):
        pass

    #! check the train method and fix the validation issue
    def train(self):
        print("\n\nTraining the model...")

        self._validate_input(train=True)

        self.augment_train_data()

        # Compute the scalers for the time domain and scale
        self.compute_scalers_transform(time_domain=True)
        # perform the same for the frequency domain data
        self.compute_scalers_transform(time_domain=False)

        print("\n\nTrain data scaling completed")
        
        if self.X_train_freq is None or self.X_train_time is None or self.y_train is None:
            raise ValueError("Training data is missing!")

        X_fft = self.X_train_freq
        X_pca = self.X_train_time
        y_train = self.y_train
        config = self.config

        n_samples = X_fft.shape[0]
        channels = X_fft.shape[1]
        fft_bins = X_fft.shape[2]
        pca_bins = X_pca.shape[2]
        steps = X_fft.shape[3]

        # Reshape inputs for the model
        X_fft = tf.reshape(X_fft, [n_samples, channels * fft_bins, steps, 1])
        X_pca = tf.reshape(X_pca, [n_samples, channels * pca_bins, steps, 1])

        # Convert the target labels to categorical if it's a classification task
        # Assuming y_train contains integer labels (e.g., 0, 1, or 2 for 3 classes)
        y_train = to_categorical(y_train, num_classes=2)  # Assuming binary classification (2 classes)

        # Get the model and early stopping callback
        model, early_stopping = MultiViewConvModel.get_model(
            config=config,
            channels=channels,
            fft_bins=fft_bins,
            pca_bins=pca_bins,
            steps=steps,
        )
        
        self.model = model

        # Train the model
        history = model.fit(
            {"time_domain_input": X_pca, "freq_domain_input": X_fft},
            {"final_output": y_train},
            epochs=config["nb_epoch"],
            verbose=1,
            batch_size=config["batch_size"],
            shuffle=True,
            callbacks=[early_stopping],  # Add early stopping
            validation_split = 0.2
        )

        print("\n\nTraining completed!")

        print("\nTrain Loss: ", history.history["loss"])
        print("Train Accuracy: ",history.history["accuracy"])
        print("\nValidation Loss: ",history.history["val_loss"])
        print("Validation Accuracy: ",history.history["val_accuracy"])
        print("Validation Precision: ",history.history["val_precision"])
        print("Validation Recall: ",history.history["val_recall"])
        
        plot_training_history(history, self.config["model_path"])

    #! implement the evaluate method
    def evaluate(self):
        self._validate_input(train=False)

    def save_model(self):
        """
        Saves the trained model to the specified file path.

        The file path is retrieved from the configuration dictionary (`self.config["model_path"]`).
        The model is serialized and saved using the `keras` library.
        File format: ".keras"

        Raises:
            KeyError: If "model_path" is not found in the configuration dictionary.
            Exception: If there is an error during the model saving process.
        """
        
        model_path = self.config["model_path"]
        model_name = self.config["name"]
        filename = model_name + ".keras"

        filepath = os.path.join(model_path, filename)

        self.model.save(filepath)
        
        print(f"\n\nModel saved to {filepath}")

    def load_model(self):
        """
        Loads a pre-trained model from the specified file path in the configuration.

        This method uses the keras library to load the model from the file path
        provided in the configuration dictionary under the key "model_path" and
        "name". The loaded model is then assigned to the instance variable `self.model`.
        File format: ".keras"

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            Exception: If there is an error during the loading of the model.
        """

        self.model = load_model_from_config(self.config)

    def _validate_input(self, train=True):
        """
        Validates the input data for training or testing.
        Parameters:
        train (bool): If True, validates the training data. If False, validates the testing data.
        Raises:
        ValueError: If the number of samples in X_train_fft, X_train_pca, and y_train are not equal (when train=True).
                    If the number of time steps in X_train_fft and X_train_pca are not equal (when train=True).
                    If the number of samples in X_test_fft, X_test_pca, and y_test are not equal (when train=False).
                    If the number of time steps in X_test_fft and X_test_pca are not equal (when train=False).
        """
        if train:
            X_train_fft = self.X_train_freq
            X_train_pca = self.X_train_time
            y_train = self.y_train

            if X_train_fft.shape[0] != X_train_pca.shape[0] or X_train_fft.shape[0] != y_train.shape[0]:
                raise ValueError(
                    "Number of samples in X_train_fft, X_train_pca, and y_train must be equal."
                )
            if X_train_fft.shape[-1] != X_train_pca.shape[-1]:
                raise ValueError(
                    "Number of time steps in X_train_fft, X_train_pca must be equal."
                )

        else:
            X_test_fft = self.X_test_freq
            X_test_pca = self.X_test_time
            y_test = self.y_test

            if X_test_fft.shape[0] != X_test_pca.shape[0] or X_test_fft.shape[0] != y_test.shape[0]:
                raise ValueError(
                    "Number of samples in X_test_fft, X_test_pca, and y_test must be equal."
                )
            if X_test_fft.shape[-1] != X_test_pca.shape[-1]:
                raise ValueError(
                    "Number of time steps in X_test_fft, X_test_pca must be equal."
                )