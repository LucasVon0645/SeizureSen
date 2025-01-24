import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import tensorflow as tf
from keras.api.utils import to_categorical, plot_model
from keras.api.models import Model
from keras.api.callbacks import ModelCheckpoint
from typing import Optional

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
        pass

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

        fit_args = {
            "x": {"time_domain_input": X_pca_train, "freq_domain_input": X_fft_train},
            "validation_data": (
            {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val},
            {"final_output": y_val},
            ),
            "epochs": config["nb_epoch"],
            "verbose": 1,
            "batch_size": config["batch_size"],
            "callbacks": [early_stopping, checkpoint],
        }

        if use_early_exits:
            print("Early exits shape:", [output.shape for output in multi_view_conv_model.outputs[1:]])
            fit_args["y"] = {
            "final_output": y_train,
            "early_exit1": y_train,
            "early_exit2": y_train,
            "early_exit3": y_train,
            "early_exit4": y_train,
            }
            fit_args["validation_data"][1].update({
            "early_exit1": y_val,
            "early_exit2": y_val,
            "early_exit3": y_val,
            "early_exit4": y_val,
            })
        else:
            fit_args["y"] = y_train
            fit_args["class_weight"] = {0: 1.0, 1: config["preictal_class_weight"]}

        history = multi_view_conv_model.fit(**fit_args)

        print("\n\nTraining completed!")

        # Save the scalers to a file
        self.save_scalers()

        print("\nTrain Loss over epochs: ", history.history["loss"])

        if use_early_exits:
            for i in range(1, 5):
                print(
                    f"Validation Accuracy for 'early_exit{i}' over epochs: ",
                    history.history[f"val_early_exit{i}_accuracy"],
                )
                print(
                "Validation Accuracy for 'final_output' over epochs: ",
                history.history["val_final_output_accuracy"],
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

    def train_fold(self, X_fft_train, X_fft_val, X_pca_train, X_pca_val, y_train, y_val, fold_index):
        """
        Train the model on a single fold.

        Parameters:
        - X_fft_train, X_fft_val: Frequency domain training and validation data.
        - X_pca_train, X_pca_val: Time domain training and validation data.
        - y_train, y_val: Training and validation labels.
        - config: Configuration dictionary.
        - fold_index: Fold index for saving checkpoints and logging.

        Returns:
        - model: The trained model.
        - metrics: Dictionary containing metrics for the fold.
        """

        # Convert target labels to categorical
        y_train = to_categorical(y_train, num_classes=2).astype("float32")
        y_val = to_categorical(y_val, num_classes=2).astype("float32")

        # Set up checkpoint path
        checkpoint_path = os.path.join(
            self.config["model_path"], f"checkpoint_fold_{fold_index}", "best_model.keras"
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        )

        # Get model and callbacks
        model, early_stopping = self.model_class.get_model(self.config)

        # Train the model
        model.fit(
            {"time_domain_input": X_pca_train, "freq_domain_input": X_fft_train},
            y_train,
            validation_data=(
                {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val},
                {"final_output": y_val},
            ),
            epochs=self.config["nb_epoch"],
            verbose=1,
            batch_size=self.config["batch_size"],
            callbacks=[early_stopping, checkpoint],
            class_weight={0: 1.0, 1: self.config["preictal_class_weight"]},  # Higher weight for minority class
        )

        # Generate predictions for the validation set
        y_val_pred = model.predict(
            {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val}
        )
        y_val_pred = np.argmax(y_val_pred, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        # Compute metrics
        metrics = classification_report(
            y_val_true, y_val_pred, target_names=["interictal", "preictal"], output_dict=True
        )
        metrics["accuracy"] = accuracy_score(y_val_true, y_val_pred)

        return model, metrics

    def train_with_cross_validation(self, n_splits=5):
        """
        Train the model using stratified k-fold cross-validation.

        Parameters:
        - n_splits: Number of folds for cross-validation.
        """

        print("\n\nTraining the model with stratified cross-validation...")
        self._validate_input(train=True)
        self.augment_train_data()

        # Compute and scale train data
        self.compute_scalers_transform(time_domain=True)
        self.compute_scalers_transform(time_domain=False)

        X_fft = self.X_train_freq.numpy()
        X_pca = self.X_train_time.numpy()
        y = self.y_train.numpy()

        n_samples = X_fft.shape[0]
        channels = X_fft.shape[1]
        fft_bins = X_fft.shape[2]
        pca_bins = X_pca.shape[2]
        steps = X_fft.shape[3]

        # Reshape inputs for the model
        X_fft = tf.reshape(X_fft, [n_samples, channels * fft_bins, steps, 1])
        X_pca = tf.reshape(X_pca, [n_samples, channels * pca_bins, steps, 1])

        # Stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_metrics = []

        for fold_index, (train_idx, val_idx) in enumerate(skf.split(X_fft, y), start=1):
            print(f"\n\nTraining fold {fold_index}/{n_splits}...")
            X_fft_numpy = X_fft.numpy()  # Convert from tensor to numpy array
            X_pca_numpy = X_pca.numpy()

            X_fft_train, X_fft_val = X_fft_numpy[train_idx], X_fft_numpy[val_idx]
            X_pca_train, X_pca_val = X_pca_numpy[train_idx], X_pca_numpy[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"Fold {fold_index} class distribution:")
            print(pd.Series(y_train).value_counts())

            _, metrics = self.train_fold(
                X_fft_train, X_fft_val, X_pca_train, X_pca_val, y_train, y_val, fold_index
            )
            all_metrics.append(metrics)

        # Print and save the cross-validation results
        self.print_and_save_cross_validation_results(all_metrics)

    def print_and_save_cross_validation_results(self, all_metrics: dict, filename="cross_validation_results.txt"):
        """
        Print the average metrics from cross-validation.

        Parameters:
        - all_metrics: List of classification reports for each fold.
        - filename: Name of the file to save the results.
        """
        
        metrics_to_print = {
            "accuracy": "Average Accuracy",
            "interictal_precision": "Average Precision (Interictal)",
            "interictal_recall": "Average Recall (Interictal)",
            "interictal_f1-score": "Average F1-Score (Interictal)",
            "preictal_precision": "Average Precision (Preictal)",
            "preictal_recall": "Average Recall (Preictal)",
            "preictal_f1-score": "Average F1-Score (Preictal)",
        }

        avg_metrics = {
            "accuracy": np.mean([metrics["accuracy"] for metrics in all_metrics]),
            "interictal_precision": np.mean(
            [metrics["interictal"]["precision"] for metrics in all_metrics]
            ),
            "interictal_recall": np.mean(
            [metrics["interictal"]["recall"] for metrics in all_metrics]
            ),
            "interictal_f1-score": np.mean(
            [metrics["interictal"]["f1-score"] for metrics in all_metrics]
            ),
            "preictal_precision": np.mean(
            [metrics["preictal"]["precision"] for metrics in all_metrics]
            ),
            "preictal_recall": np.mean(
            [metrics["preictal"]["recall"] for metrics in all_metrics]
            ),
            "preictal_f1-score": np.mean(
            [metrics["preictal"]["f1-score"] for metrics in all_metrics]
            ),
        }

        print("\n\nCross-Validation Results:")
        with open(os.path.join(self.config["model_path"], filename), "w", encoding="utf-8") as f:
            for key, description in metrics_to_print.items():
                print(f"{description}: {avg_metrics[key]:.4f}")
                f.write(f"{description}: {avg_metrics[key]:.4f}\n")

    def evaluate(self, use_early_exits=False):
        """
        Evaluate the multi-view convolutional neural network model.
        This method evaluates the model using the test data.
        The model is loaded from the configuration settings, and the test data
        is scaled using the scalers learned from the training data.
        The classification report is generated for the test data.
        """

        self._validate_input(train=False)
        
        if self.X_test_freq is None or self.X_test_time is None or self.y_test is None:
            raise ValueError("Testing data is missing!")

        # Scale the test data using the scalers learned from the training data
        X_pca = scale_across_time_tf(self.X_test_time, self.scalers_time)
        X_fft = scale_across_time_tf(self.X_test_freq, self.scalers_freq)
        y = self.y_test

        print("\n\nEvaluating the model...")

        # Get the model
        multi_view_conv_model = self.model

        n_samples = X_fft.shape[0]
        channels = X_fft.shape[1]
        fft_bins = X_fft.shape[2]
        pca_bins = X_pca.shape[2]
        steps = X_fft.shape[3]
        config = self.config
        
        use_early_exits = config.get("use_early_exits", False)

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

    def train_full_dataset(self):
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

        X_fft_train = self.X_train_freq
        X_pca_train = self.X_train_time
        y_train = self.y_train
        config = self.config
        
        use_early_exits = config.get("use_early_exits", False)

        n_samples_train = X_fft_train.shape[0]
        channels = X_fft_train.shape[1]
        fft_bins = X_fft_train.shape[2]
        pca_bins = X_pca_train.shape[2]
        steps = X_fft_train.shape[3]

        # Reshape inputs for the model
        # The last dimension of the input shape is 1 because the input data
        # is single-channel (like grayscale image)
        X_fft_train = tf.reshape(X_fft_train, [n_samples_train, channels * fft_bins, steps, 1])
        X_pca_train = tf.reshape(X_pca_train, [n_samples_train, channels * pca_bins, steps, 1])

        # Scale the test data using the scalers learned from the training data
        X_fft_val = scale_across_time_tf(self.X_test_freq, self.scalers_freq)
        X_pca_val = scale_across_time_tf(self.X_test_time, self.scalers_time)
        y_val = self.y_test
        
        n_samples_test = X_fft_val.shape[0]
        
        X_fft_val = tf.reshape(X_fft_val, [n_samples_test, channels * fft_bins, steps, 1])
        X_pca_val = tf.reshape(X_pca_val, [n_samples_test, channels * pca_bins, steps, 1])

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
        y_train = to_categorical(y_train, num_classes=2).numpy()
        y_val = to_categorical(y_val, num_classes=2).numpy()

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

        fit_args = {
            "x": {"time_domain_input": X_pca_train, "freq_domain_input": X_fft_train},
            "validation_data": (
            {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val},
            {"final_output": y_val},
            ),
            "epochs": config["nb_epoch"],
            "verbose": 1,
            "batch_size": config["batch_size"],
            "callbacks": [early_stopping, checkpoint],
        }

        if use_early_exits:
            print("Early exits shape:", [output.shape for output in multi_view_conv_model.outputs[1:]])
            fit_args["y"] = {
            "final_output": y_train,
            "early_exit1": y_train,
            "early_exit2": y_train,
            "early_exit3": y_train,
            "early_exit4": y_train,
            }
            fit_args["validation_data"][1].update({
            "early_exit1": y_val,
            "early_exit2": y_val,
            "early_exit3": y_val,
            "early_exit4": y_val,
            })
        else:
            fit_args["y"] = y_train
            fit_args["class_weight"] = {0: 1.0, 1: config["preictal_class_weight"]}

        history = multi_view_conv_model.fit(**fit_args)

        print("\n\nTraining completed!")

        # Save the scalers to a file
        self.save_scalers()

        print("\nTrain Loss over epochs: ", history.history["loss"])

        if use_early_exits:
            for i in range(1, 5):
                print(
                    f"Validation Accuracy for 'early_exit{i}' over epochs: ",
                    history.history[f"val_early_exit{i}_accuracy"],
                )
                print(
                "Validation Accuracy for 'final_output' over epochs: ",
                history.history["val_final_output_accuracy"],
                )
        else:
            print("Train Accuracy over epochs: ", history.history["accuracy"])

        print("\n\nModel Validation")

        print("\nValidation Loss over epochs: ", history.history["val_loss"], "\n")

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
                    f"Number of samples in X_test_fft, X_test_pca, and y_test must be equal. But got: {(X_test_fft.shape[0], X_test_pca.shape[0], y_test.shape[0])}"
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
