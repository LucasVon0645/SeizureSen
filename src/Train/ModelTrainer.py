import os
import sys
import joblib
import pandas as pd
import matplotlib

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN

import tensorflow as tf
from keras.api.utils import to_categorical, plot_model
from keras.api.models import Model
from keras.api.callbacks import ModelCheckpoint
from typing import Optional
from src.Preprocessing.utils import overlap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Train.utils import (
    plot_training_history,
    save_confusion_matrix,
    save_model_scores,
)
from src.Preprocessing.utils import (
    load_preprocessed_data,
    transform_to_tensor,
    scale_across_time_tf,
)

from src.Model.utils import (
    load_model_from_config,
    load_config,
    load_scalers_from_config,
)


class ModelTrainer:
    """
    A class to train and evaluate a multi-view convolutional neural network model
    for EEG data analysis.
    This class loads the configuration settings, preprocessed EEG data, and
    trains the model using the training data.
    The trained model is then evaluated using the test data, and the results
    are saved to a specified file path.
    """

    def __init__(self, cfg_path: str, data_directory: str, model_class: type):
        self.data_path = data_directory
        self.config_path = cfg_path

        self.config = load_config(cfg_path)

        self.model: Optional[Model] = None
        self.model_class = model_class  # Store the model class for later use

        # Freq Domain Tensorflow Tensors (n_samples, n_channels, fft_bins, steps)
        self.X_train_freq: Optional[tf.Tensor] = None
        self.X_test_freq: Optional[tf.Tensor] = None

        # Time Domain Tensorflow Tensors (n_samples, n_channels, pca_bins, steps)
        self.X_train_time: Optional[tf.Tensor] = None
        self.X_test_time: Optional[tf.Tensor] = None

        # Label Tensors (n_samples,)
        self.y_train: Optional[tf.Tensor]= None
        self.y_test: Optional[tf.Tensor] = None

        # Scalers for both time and frequency domain
        # Lists of dictionaries {"mean": float, "std": float}
        # One dictionary per channel
        self.scalers_time: Optional[list[dict]] = None
        self.scalers_freq: Optional[list[dict]] = None

        matplotlib.use(
            "Agg"
        )  # Use the Agg backend to avoid displaying plots during execution

        model_path = self.config["model_path"]

        os.makedirs(model_path, exist_ok=True)

    def load_data(self, file_names_dict: Optional[dict] = None):

        #def load_data(self, file_names_dict=dict | None):
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

        if file_names_dict is None:
            file_names_dict = {
                "freq_train": "freq_domain_train.npz",
                "freq_test": "freq_domain_test.npz",
                "time_train": "time_domain_train.npz",
                "time_test": "time_domain_test.npz",
            }

        # Load preprocessed slices of eeg data for train and evaluation

        data_train_freq = load_preprocessed_data(
            data_dir, file_names_dict["freq_train"]
        )
        data_test_freq = load_preprocessed_data(data_dir, file_names_dict["freq_test"])

        data_train_time = load_preprocessed_data(
            data_dir, file_names_dict["time_train"]
        )
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

    def compute_scalers_transform(self, time_domain=True):
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
                [sample_num * time_step_num, bin_num],
            )

            # Compute the mean and standard deviation for the channel
            mean = tf.reduce_mean(channel_data, axis=0)
            std = tf.math.reduce_std(channel_data, axis=0)

            # Standardize the data
            standardized_channel_data = (channel_data - mean) / std

            # Store the mean and std for this channel (optional, if you need to use them later)
            scalers.append({"mean": mean, "std": std})

            # Reshape and transpose the standardized data back to the original dimensions
            standardized_channel_data = tf.transpose(
                tf.reshape(
                    standardized_channel_data, [sample_num, time_step_num, bin_num]
                ),
                perm=[0, 2, 1],
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
        filename = "feature_scalers.pkl"

        filepath = os.path.join(model_path, filename)

        scalers = {
            "time_domain": self.scalers_time,
            "frequency_domain": self.scalers_freq,
        }

        # Save the dictionary to a file using joblib
        joblib.dump(scalers, filepath)

    #! implement the augment_train_data method
    def augment_train_data(self):
        # Reshape data for SMOTE
        X_train_freq_reshaped = self.X_train_freq.numpy().reshape((self.X_train_freq.shape[0], -1))
        X_train_time_reshaped = self.X_train_time.numpy().reshape((self.X_train_time.shape[0], -1))
        y_train = self.y_train.numpy()

        # Apply SMOTE to augment the preictal data
        smote = SMOTE(sampling_strategy='minority')
        X_train_freq_resampled, y_train_resampled = smote.fit_resample(X_train_freq_reshaped, y_train)
        X_train_time_resampled, _ = smote.fit_resample(X_train_time_reshaped, y_train)

        # Reshape back to original dimensions
        self.X_train_freq = tf.convert_to_tensor(X_train_freq_resampled.reshape((-1, *self.X_train_freq.shape[1:])))
        self.X_train_time = tf.convert_to_tensor(X_train_time_resampled.reshape((-1, *self.X_train_time.shape[1:])))
        self.y_train = tf.convert_to_tensor(y_train_resampled)




    def train(self, use_early_exits=False):
        """
        Train the multi-view convolutional neural network model.
        This method trains and validates the model using the training data.
        All hyperparameters, the number of epochs and the batch size are specified
        in the configuration settings.
        The best model is saved to a file using the ModelCheckpoint callback in a directory
        "checkpoint" within the model directory specified in the configuration.
        """

        print("\n\nTraining the model...")

        self._validate_input(train=True)

        self.augment_train_data()

        # Compute the scalers for the time domain and scale
        self.compute_scalers_transform(time_domain=True)
        # Perform the same for the frequency domain data
        self.compute_scalers_transform(time_domain=False)

        print("\n\nTrain data scaling completed")

        if (
            self.X_train_freq is None
            or self.X_train_time is None
            or self.y_train is None
        ):
            raise ValueError("Training data is missing!")

        X_fft = self.X_train_freq
        X_pca = self.X_train_time
        y = self.y_train
        config = self.config

        n_samples = X_fft.shape[0]
        channels = X_fft.shape[1]
        fft_bins = X_fft.shape[2]
        pca_bins = X_pca.shape[2]
        steps = X_fft.shape[3]

        # Reshape inputs for the model
        # The last dimension of the input shape is 1 because the input data
        # is single-channel (like grayscale image)
        X_fft = tf.reshape(X_fft, [n_samples, channels * fft_bins, steps, 1])
        X_pca = tf.reshape(X_pca, [n_samples, channels * pca_bins, steps, 1])

        # Split the data into training and validation sets
        X_fft_train, X_fft_val, X_pca_train, X_pca_val, y_train, y_val = (
            train_test_split(
                X_fft.numpy(),
                X_pca.numpy(),
                y.numpy(),
                test_size=0.2,
                random_state=42,
                stratify=y,
                shuffle=True,
            )
        )

        # Print the class distribution for the training
        train_dist = (
            pd.Series(y_train).map({0: "interictal", 1: "preictal"}).value_counts()
        )
        print("\n\nTraining data class distribution:")
        print(train_dist)
        # Print the class distribution for the validation
        val_dist = pd.Series(y_val).map({0: "interictal", 1: "preictal"}).value_counts()
        print("\n\nValidation data class distribution:")
        print(val_dist, "\n\n")

        # Convert the target labels to categorical, as it's a classification task
        # The labels are one-hot encoded (e.g., [0, 1] for preictal and [1, 0] for interictal)
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")

        # Add model checkpoint callback to automatically save the best model
        model_path = self.config["model_path"]
        checkpoint_path = os.path.join(model_path, "checkpoint", "best_model.keras")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        )

        # Create non-trained model object and get the early stopping callback
        multi_view_conv_model, early_stopping = self.model_class.get_model(config)

        self.model = multi_view_conv_model

        print("Fitting the model...")
        print("Final output shape:", multi_view_conv_model.output[0].shape)

        if not use_early_exits:
            # Train the model
            history = multi_view_conv_model.fit(
                {"time_domain_input": X_pca_train, "freq_domain_input": X_fft_train},
                y_train,
                validation_data=(
                    {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val},
                    {"final_output": y_val},
                ),
                epochs=config["nb_epoch"],
                verbose=1,
                batch_size=config["batch_size"],
                callbacks=[
                    early_stopping,
                    checkpoint,
                ],  # Add early stopping and checkpoint callbacks
            )

        else:
            multi_view_conv_model.compile(
                optimizer="adam",
                loss={
                    "final_output": "categorical_crossentropy",
                    "early_exit1": "categorical_crossentropy",
                    "early_exit2": "categorical_crossentropy",
                    "early_exit3": "categorical_crossentropy",
                    "early_exit4": "categorical_crossentropy",
                },
                # Weights for the losses of the different outputs
                loss_weights={
                    "final_output": 1.0,
                    "early_exit1": 0.5,
                    "early_exit2": 0.7,
                    "early_exit3": 0.8,
                    "early_exit4": 1.0,
                },
                metrics={
                    "final_output": ["accuracy"],
                    "early_exit1": ["accuracy"],
                    "early_exit2": ["accuracy"],
                    "early_exit3": ["accuracy"],
                    "early_exit4": ["accuracy"],
                },
            )
            
            print("Early exits shape:", [output.shape for output in multi_view_conv_model.outputs[1:]])

            history = multi_view_conv_model.fit(
                {"time_domain_input": X_pca_train, "freq_domain_input": X_fft_train},
                {
                    "final_output": y_train,
                    "early_exit1": y_train,
                    "early_exit2": y_train,
                    "early_exit3": y_train,
                    "early_exit4": y_train,
                },
                validation_data=(
                    {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val},
                    {
                        "final_output": y_val,
                        "early_exit1": y_val,
                        "early_exit2": y_val,
                        "early_exit3": y_val,
                        "early_exit4": y_val,
                    },
                ),
                epochs=config["nb_epoch"],
                batch_size=config["batch_size"],
                callbacks=[early_stopping, checkpoint],
            )

        print("\n\nTraining completed!")

        # Save the scalers to a file
        self.save_scalers()

        print("\nTrain Loss over epochs: ", history.history["loss"])

        if use_early_exits:
            print(
                "\nValidation Accuracy for 'final_output' over epochs: ",
                history.history["val_final_output_accuracy"],
            )
            print(
                "Validation Accuracy for 'early_exit1' over epochs: ",
                history.history["val_early_exit1_accuracy"],
            )
            print(
                "Validation Accuracy for 'early_exit2' over epochs: ",
                history.history["val_early_exit2_accuracy"],
            )
            print(
                "Validation Accuracy for 'early_exit3' over epochs: ",
                history.history["val_early_exit3_accuracy"],
            )
            print(
                "Validation Accuracy for 'early_exit4' over epochs: ",
                history.history["val_early_exit4_accuracy"],
            )
        else:
            print("Train Accuracy over epochs: ", history.history["accuracy"])

        print("\n\nModel Validation")

        print("\nValidation Loss over epochs: ", history.history["val_loss"])
        
        print("")

        y_pred = multi_view_conv_model.predict(
            {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val}
        )

        # Get the classification report for the validation data and save it
        self._get_classification_report(y_val, y_pred, use_early_exits, suffix="val")

        # Save the training history
        plot_training_history(history, config["model_path"])

        # Uncomment the two following lines only if you have graphviz installed!
        filepath = os.path.join(
            config["model_path"], config["name"] + "_architecture.png"
        )
        plot_model(
            multi_view_conv_model,
            to_file=filepath,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
        )

    def evaluate(self, use_early_exits=False):
        """
        Evaluate the multi-view convolutional neural network model.
        This method evaluates the model using the test data.
        The model is loaded from the configuration settings, and the test data
        is scaled using the scalers learned from the training data.
        The classification report is generated for the test data.
        """

        self._validate_input(train=False)

        # Scale the test data using the scalers learned from the training data
        scale_across_time_tf(self.X_test_time, self.scalers_time)
        scale_across_time_tf(self.X_test_freq, self.scalers_freq)

        print("\n\nEvaluating the model...")

        if self.X_test_freq is None or self.X_test_time is None or self.y_test is None:
            raise ValueError("Testing data is missing!")

        # Get the test data
        X_fft = self.X_test_freq
        X_pca = self.X_test_time
        y = self.y_test

        # Get the model
        multi_view_conv_model = self.model

        n_samples = X_fft.shape[0]
        channels = X_fft.shape[1]
        fft_bins = X_fft.shape[2]
        pca_bins = X_pca.shape[2]
        steps = X_fft.shape[3]

        # Reshape inputs for the model
        # The last dimension of the input shape is 1 because the input data
        # is single-channel (like grayscale image)
        X_fft = tf.reshape(X_fft, [n_samples, channels * fft_bins, steps, 1])
        X_pca = tf.reshape(X_pca, [n_samples, channels * pca_bins, steps, 1])

        # Convert the target labels to categorical
        y_test = to_categorical(y, num_classes=2).numpy()

        # Evaluate the model
        y_pred = multi_view_conv_model.predict(
            {"time_domain_input": X_pca, "freq_domain_input": X_fft}
        )

        # Get the classification report for the test data and save it
        self._get_classification_report(y_test, y_pred, use_early_exits, suffix="eval")

        print("\nEvaluation completed!")

    def load_model(self):
        """
        Loads the machine learning model based on the provided configuration.
        This method initializes the model by loading it from the configuration
        specified in `self.config`. Additionally, it loads the necessary scalers
        required for preprocessing the data.
        Returns:
            None
        """

        self.model = load_model_from_config(self.config)
        self.load_scalers()

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
        config = self.config
        if train:
            X_train_fft = self.X_train_freq
            X_train_pca = self.X_train_time
            y_train = self.y_train

            if (
                X_train_fft.shape[1] != config["channels"]
                or X_train_pca.shape[1] != config["channels"]
            ):
                raise ValueError(
                    "Number of channels in X_train_fft and X_train_pca must match the configuration."
                )

            if (
                X_train_fft.shape[3] != config["model_time_steps"]
                or X_train_pca.shape[3] != config["model_time_steps"]
            ):
                raise ValueError(
                    "Number of time steps in X_train_fft and X_train_pca must match the configuration."
                )

            if (
                X_train_fft.shape[2] != config["fft_bins"]
                or X_train_pca.shape[2] != config["pca_bins"]
            ):
                raise ValueError(
                    "Number of FFT bins in X_train_fft and PCA bins in X_train_pca must match the configuration."
                )

            if (
                X_train_fft.shape[0] != X_train_pca.shape[0]
                or X_train_fft.shape[0] != y_train.shape[0]
            ):
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

            if (
                X_test_fft.shape[1] != config["channels"]
                or X_test_pca.shape[1] != config["channels"]
            ):
                raise ValueError(
                    "Number of channels in X_test_fft and X_test_pca must match the configuration."
                )

            if (
                X_test_fft.shape[3] != config["model_time_steps"]
                or X_test_pca.shape[3] != config["model_time_steps"]
            ):
                raise ValueError(
                    "Number of time steps in X_test_fft and X_test_pca must match the configuration."
                )

            if (
                X_test_fft.shape[2] != config["fft_bins"]
                or X_test_pca.shape[2] != config["pca_bins"]
            ):
                raise ValueError(
                    "Number of FFT bins in X_test_fft and PCA bins in X_test_pca must match the configuration."
                )

            if (
                X_test_fft.shape[0] != X_test_pca.shape[0]
                or X_test_fft.shape[0] != y_test.shape[0]
            ):
                raise ValueError(
                    "Number of samples in X_test_fft, X_test_pca, and y_test must be equal."
                )
            if X_test_fft.shape[-1] != X_test_pca.shape[-1]:
                raise ValueError(
                    "Number of time steps in X_test_fft, X_test_pca must be equal."
                )

    def _get_classification_report(
        self, y_true, y_pred, early_exits_used=False, suffix=""
    ):
        """
        Get the classification report for the model.
        The plot is saved in the model directory.
        Parameters:
            y_true (np.ndarray): The true labels in categorical format (matrix like).
            y_pred (np.ndarray): The predicted labels in categorical format (matrix like).
            suffix (str): The suffix to add to the file name.
        """
        # Binarize predictions (e.g., threshold = 0.5 for binary classification)
        if early_exits_used:
            # Only the final_output is considered for the classification report
            y_pred_final = y_pred[0]
            y_pred_classes = (y_pred_final[:, 1] > 0.5).astype(int)
        else:
            y_pred_classes = (y_pred[:, 1] > 0.5).astype(int)

        # Binarize true labels (e.g., threshold = 0.5 for binary classification)
        y_true_classes = y_true[:, 1].astype(int)

        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=["Interictal", "Preictal"],
            zero_division=0,
        )

        print("\n\nClassification Report")
        print(report)

        model_path = self.config["model_path"]

        filename = f"classification_report_{suffix}.txt"
        save_model_scores(report, os.path.join(model_path, filename))

        filename = f"confusion_matrix_{suffix}.png"
        save_confusion_matrix( 
            y_true_classes,
            y_pred_classes,
            os.path.join(model_path, filename),
        )
