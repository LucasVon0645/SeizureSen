from typing import Optional
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN

import tensorflow as tf
from keras.api.utils import to_categorical, plot_model
from keras.api.models import Model
from keras.api.callbacks import ModelCheckpoint
from numpy import ndarray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Train.utils import (
    plot_training_history,
    save_confusion_matrix,
    save_model_scores,
    plot_roc_curve,
    print_and_save_cross_validation_results,
    get_optimal_threshold,
    save_threshold
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

from src.Model.MultiViewConvModel import MultiViewConvModel
from src.Model.MultiViewConvModel_v2 import MultiViewConvModel_v2
from src.Model.MultiViewConvModelAttention import MultiViewConvModelWithAttention
from src.Model.MultiViewConvModelBatchNorm import MultiViewConvModelWithBatchNorm


class ModelTrainer:
    """
    A class to train and evaluate a multi-view convolutional neural network model
    for EEG data analysis.
    This class loads the configuration settings, preprocessed EEG data, and
    trains the model using the training data.
    The trained model is then evaluated using the test data, and the results
    are saved to a specified file path.
    """

    def __init__(self, cfg_path: str, data_directory: str, model_class: Optional[type] = None):
        self.data_path = data_directory
        self.config_path = cfg_path

        self.config = load_config(cfg_path)

        self.model: Optional[Model] = None
        if model_class is not None:
            self.model_class = model_class  # Store the model class for later use
        else:
            self._set_model_class()

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
        
        self.augmentation_strategy = self.config.get("augmentation_strategy", None)
        self.preictal_class_weight = self.config.get("preictal_class_weight", None)

        matplotlib.use(
            "Agg"
        )  # Use the Agg backend to avoid displaying plots during execution

        model_path = self.config["model_path"]

        os.makedirs(model_path, exist_ok=True)
        
        self._save_model_config()
        
        self.optimal_threshold = None

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

    def save_scalers(self, filename="feature_scalers.pkl"):
        """
        Save scalers to a file.
        """

        model_path = self.config["model_path"]

        filepath = os.path.join(model_path, filename)

        scalers = {
            "time_domain": self.scalers_time,
            "frequency_domain": self.scalers_freq,
        }

        # Save the dictionary to a file using joblib
        joblib.dump(scalers, filepath)

    def augment_train_data(self, X_train_time: ndarray, X_train_freq: ndarray, y_train: ndarray, strategy="SMOTE"):
        """
        Augment the training data using SMOTE or ADASYN.
        This method augments the preictal data using the specified strategy
        (either SMOTE or ADASYN) to balance the class distribution.
        The augmented data is reshaped back to the original dimensions.
        Parameters:
            - strategy (str): The strategy to use for data augmentation.
                              Either 'SMOTE' or 'ADASYN'.
        Returns:
            - X_train_time_resampled (np.ndarray): The resampled time domain data.
            - X_train_freq_resampled (np.ndarray): The resampled frequency domain data.
            - y_train_resampled (np.ndarray): The resampled labels.
        """
        print("\n\nInitial shape of the training data (X_time, X_freq, y): ", X_train_time.shape, X_train_freq.shape, y_train.shape)
        print("\n\nOriginal training data class distribution:")
        print(pd.Series(y_train).map({0: "interictal", 1: "preictal"}).value_counts())

        # Check shapes of input arrays
        assert X_train_freq.shape[0] == X_train_time.shape[0] == y_train.shape[0], (
            "X_train_freq, X_train_time, and y_train must have the same number of samples."
        )
        assert strategy in ["SMOTE", "ADASYN"], (
            f"Invalid strategy '{strategy}'. Choose 'SMOTE' or 'ADASYN'."
        )

        # Reshape data for SMOTE
        X_train_freq_reshaped = X_train_freq.reshape((X_train_freq.shape[0], -1))
        X_train_time_reshaped = X_train_time.reshape((X_train_time.shape[0], -1))

        # Get the number of features in the frequency domain data after reshaping
        num_features_in_X_train_freq = X_train_freq_reshaped.shape[1]

        # Combine the time and frequency domain data
        # Rows represent samples and columns represent features
        X_train_combined = np.concatenate((X_train_freq_reshaped, X_train_time_reshaped), axis=1)

        if strategy == "ADASYN":
            print("\n\nApplying ADASYN to augment the preictal data...")
            # Apply ADASYN to augment the preictal data
            adasyn = ADASYN(sampling_strategy='minority', random_state=42)
            X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_combined, y_train)
        else:
            print("\n\nApplying SMOTE to augment the preictal data...")
            # Apply SMOTE to augment the preictal data
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

        # Reshape the resampled data back to the original dimensions, separating the freq and time features
        X_train_freq_resampled = X_train_resampled[:, :num_features_in_X_train_freq]
        X_train_time_resampled = X_train_resampled[:, num_features_in_X_train_freq:]

        # Reshape back to original dimensions        
        X_train_freq_resampled = X_train_freq_resampled.reshape((-1, *X_train_freq.shape[1:]))
        X_train_time_resampled = X_train_time_resampled.reshape((-1, *X_train_time.shape[1:]))

        return X_train_time_resampled, X_train_freq_resampled, y_train_resampled

    def train(self, threshold_tunning=False):
        """
        Train the multi-view convolutional neural network model.
        This method trains and validates the model using the training data.
        All hyperparameters, the number of epochs and the batch size are specified
        in the configuration settings.
        The best model is saved to a file using the ModelCheckpoint callback in a directory
        "checkpoint" within the model directory specified in the configuration.
        If threshold_tunning is True, the optimal threshold for classification is computed based on the validation data.
        Validation results are also obtained using the optimal threshold.
        Args:
            threshold_tunning (bool): If True, get the optimal threshold for classification.
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
        
        # Augment the training data if a strategy is provided
        if self.augmentation_strategy is not None:
            X_pca_train, X_fft_train, y_train = self.augment_train_data(X_pca_train, X_fft_train, y_train, self.augmentation_strategy)

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

        use_early_exits = config.get("use_early_exits", False)

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
            if self.preictal_class_weight is not None:
                print("Using class weights for the preictal class: ", config["preictal_class_weight"])
                fit_args["class_weight"] = {0: 1.0, 1: config["preictal_class_weight"]}

        print("Fitting the model...")
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
        
        if use_early_exits:
            y_pred = y_pred[0]

        # Get the classification report for the validation data and save it
        self._get_classification_report(y_val, y_pred, suffix="val")

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
        
        print("\n\nModel architecture saved to: ", filepath)
        
        if threshold_tunning:
            # Get the threshold that maximizes the F1 score
            self.optimal_threshold = get_optimal_threshold(y_val, y_pred)
            
            print(f"\nOptimal threshold: {self.optimal_threshold}")
            
            print("\n\nModel Validation with optimal threshold")
            
            # Get the classification report for the test data after threshold tunning
            self._get_classification_report(y_val, y_pred, suffix="threshold_tunning_val", threshold=self.optimal_threshold)

            # Save the threshold to a file
            save_threshold(self.config["model_path"], self.optimal_threshold, "F1-score")
            
        print("\n\nTraining completed!")

    def train_fold(self, X_fft_train: ndarray, X_fft_val: ndarray, X_pca_train: ndarray, X_pca_val: ndarray, y_train: ndarray, y_val: ndarray, fold_index):
        """
        Train the model on a single fold.

        Parameters:
        - X_fft_train, X_fft_val: Frequency domain training and validation data.
        - X_pca_train, X_pca_val: Time domain training and validation data.
        - y_train, y_val: Training and validation labels.
        - config: Configuration dictionary.
        - fold_index: Fold index for saving checkpoints and logging.
        - augmentation_strategy: Data augmentation strategy to use. Default is None (No augmentation). Possible values are "SMOTE" and "ADASYN".

        Returns:
        - model: The trained model.
        - metrics: Dictionary containing metrics for the fold.
        """
        config = self.config
        use_early_exits = config.get("use_early_exits", False)
        
        # Augment the training data if a strategy is provided
        if self.augmentation_strategy is not None:
            X_pca_train, X_fft_train, y_train = self.augment_train_data(X_pca_train, X_fft_train, y_train, self.augmentation_strategy)
        
        print(f"Distribution of the training data after augmentation in folder {fold_index}:")
        print(pd.Series(y_train).map({0: "interictal", 1: "preictal"}).value_counts())

        # Convert target labels to categorical
        y_train = to_categorical(y_train, num_classes=2).astype("float32")
        y_val = to_categorical(y_val, num_classes=2).astype("float32")

        # Get model and callbacks
        model, early_stopping = self.model_class.get_model(self.config)
        
        fit_args = {
            "x": {"time_domain_input": X_pca_train, "freq_domain_input": X_fft_train},
            "validation_data": (
            {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val},
            {"final_output": y_val},
            ),
            "epochs": config["nb_epoch"],
            "verbose": 1,
            "batch_size": config["batch_size"],
            "callbacks": [early_stopping],
        }

        if use_early_exits:
            print("Early exits shape:", [output.shape for output in model.outputs[1:]])
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
            if self.preictal_class_weight is not None:
                print("Using class weights for the preictal class: ", config["preictal_class_weight"])
                fit_args["class_weight"] = {0: 1.0, 1: config["preictal_class_weight"]}

        print(f"Fitting the model in folder {fold_index}...")
        model.fit(**fit_args)

        # Generate predictions for the validation set
        y_val_pred = model.predict(
            {"time_domain_input": X_pca_val, "freq_domain_input": X_fft_val}
        )
        
        if use_early_exits:
            y_val_pred = y_val_pred[0]
        
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
                X_fft_train, X_fft_val,
                X_pca_train, X_pca_val,
                y_train, y_val,
                fold_index
            )
            
            all_metrics.append(metrics)

        # Print and save the cross-validation results
        print_and_save_cross_validation_results(all_metrics, self.config["model_path"])

    def evaluate(self, save_test_pred = False, use_optimal_threshold = False):
        """
        Evaluate the multi-view convolutional neural network model.
        This method evaluates the model using the test data.
        The model is loaded from the configuration settings, and the test data
        is scaled using the scalers learned from the training data.
        The classification report is generated for the test data.
        Args:
            save_test_pred (bool): If True, save the model predictions to a file.
            use_optimal_threshold (bool): If True, use the optimal threshold for classification as well.
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

        if use_early_exits:
            y_pred = y_pred[0] # Get the final output

        # Get the classification report for the test data and save it
        self._get_classification_report(y_test, y_pred, suffix="eval")

        plot_roc_curve(y_test, y_pred, config["model_path"], "roc_curve_eval.png")

        if save_test_pred:
            # Save the model predictions
            self._save_predictions_to_file(y_pred, y_test, file_name="predictions_eval.csv")

        if use_optimal_threshold:
            if self.optimal_threshold is None:
                raise ValueError(
                    "Optimal threshold is missing! Run the training with get_optimal_threshold=True to get the optimal threshold."
                )

            print("\n\nModel Evaluation with optimal threshold")

            # Get the classification report for the test data after threshold tunning
            self._get_classification_report(y_test, y_pred, suffix="threshold_tunning_eval",
                                            threshold=self.optimal_threshold)
            # Save the model predictions with the optimal threshold
            self._save_predictions_to_file(y_pred, y_test, file_name="predictions_threshold_tunning_eval.csv",
                                           threshold=self.optimal_threshold)

        print("\nEvaluation completed!")

    def train_full_dataset(self, save_test_pred = False):
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
        X_fft_train = tf.reshape(X_fft_train, [n_samples_train, channels * fft_bins, steps, 1]).numpy()
        X_pca_train = tf.reshape(X_pca_train, [n_samples_train, channels * pca_bins, steps, 1]).numpy()
        y_train = y_train.numpy()
        
        if self.augmentation_strategy is not None:
            X_pca_train, X_fft_train, y_train = self.augment_train_data(X_pca_train, X_fft_train, y_train, self.augmentation_strategy)

        # Scale the test data using the scalers learned from the training data
        X_fft_val = scale_across_time_tf(self.X_test_freq, self.scalers_freq)
        X_pca_val = scale_across_time_tf(self.X_test_time, self.scalers_time)
        y_val = self.y_test
        
        n_samples_test = X_fft_val.shape[0]
        
        X_fft_val = tf.reshape(X_fft_val, [n_samples_test, channels * fft_bins, steps, 1]).numpy()
        X_pca_val = tf.reshape(X_pca_val, [n_samples_test, channels * pca_bins, steps, 1]).numpy()
        y_val = y_val.numpy()
    
        # Print the class distribution for the training
        print("\n\nTraining data class distribution:")
        print(pd.Series(y_train).map({0: "interictal", 1: "preictal"}).value_counts())
        # Print the class distribution for the validation
        print("\n\nValidation data class distribution:")
        print(pd.Series(y_val).map({0: "interictal", 1: "preictal"}).value_counts(), "\n\n")

        # Convert the target labels to categorical, as it's a classification task
        # The labels are one-hot encoded (e.g., [0, 1] for preictal and [1, 0] for interictal)
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")

        # Add model checkpoint callback to automatically save the best model
        model_path = self.config["model_path"]
        checkpoint_path = os.path.join(model_path, "checkpoint_full_dataset", "best_model.keras")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        )

        # Create non-trained model object and get the early stopping callback
        multi_view_conv_model, early_stopping = self.model_class.get_model(config)

        self.model = multi_view_conv_model

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
            if self.preictal_class_weight is not None:
                print("Using class weights for the preictal class: ", config["preictal_class_weight"])
                fit_args["class_weight"] = {0: 1.0, 1: config["preictal_class_weight"]}

        print("Fitting the model...")
        history = multi_view_conv_model.fit(**fit_args)

        print("\n\nTraining completed!")

        # Save the scalers to a file
        self.save_scalers("feature_scalers_full_dataset.pkl")

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
        
        if use_early_exits:
            y_pred = y_pred[0]
        
        if save_test_pred:
            # Save the model predictions
            self._save_predictions_to_file(y_pred, y_val, file_name="test_prediction_full_dataset.csv")

        # Get the classification report for the validation data and save it
        self._get_classification_report(y_val, y_pred, "eval_full_dataset")

        plot_roc_curve(y_val, y_pred, config["model_path"], "roc_curve_full_dataset_eval.png")

        # Save the training history
        plot_training_history(history, config["model_path"], suffix="_full_dataset")

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

    def load_model(self,  weights_path = None, scalers_path = None):
        """
        Loads the machine learning model based on the provided configuration.
        This method initializes the model by loading it from the configuration
        specified in `self.config`. Additionally, it loads the necessary scalers
        required for preprocessing the data.
        Returns:
            None
        """
        if weights_path is None and scalers_path is None:
            self.model = load_model_from_config(self.config, self.model_class)
            self.load_scalers()
        else:
            self.model, _ = self.model_class.get_model(self.config)
            self.model.load_weights(weights_path)
            
             # Load the scalers from the file
            scalers = joblib.load(scalers_path)
            # Access the time domain and frequency domain scalers
            self.scalers_time = scalers["time_domain"]
            self.scalers_freq = scalers["frequency_domain"]

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
        self, y_true: ndarray, y_pred: ndarray, threshold=0.5, suffix=""
    ):
        """
        Get the classification report for the model.
        The plot is saved in the model directory.
        Parameters:
            y_true (np.ndarray): The true labels in categorical format (matrix like).
            y_pred (np.ndarray): The predicted labels in categorical format (matrix like).
            threshold (float): The threshold to use for binarizing the predictions.
            suffix (str): The suffix to add to the file name.
        """
        # Binarize predictions (e.g., threshold = 0.5 for binary classification)
        y_pred_prob = y_pred[:, 1]
        y_pred_classes = (y_pred_prob > threshold).astype(int)

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

    def _save_model_config(self):
        """
        Save the model configuration to a file.
        """
        model_path = self.config["model_path"]
        filename = "model_config.json"

        filepath = os.path.join(model_path, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def _save_predictions_to_file(self, y_pred: ndarray, y_test: ndarray, file_name, threshold=0.5):
        """
        Save a DataFrame with predicted labels, true labels, and prediction
        probabilities for the 'preictal' class.

        Parameters:
        - y_pred (ndarray): Array with predicted probabilities for each class (shape: [n_samples, n_classes]).
        - y_test (ndarray): One-hot encoded array of true labels (shape: [n_samples, n_classes]).
        - file_name (str): Name of the file where the DataFrame will be saved.

        Returns:
        - None
        """
        # Determine predicted labels
        pred_labels = (y_pred[:, 1] > threshold).astype(int)

        # Get the true labels from one-hot encoding
        true_labels = np.argmax(y_test, axis=1)

        # Get predicted probabilities for the 'preictal' class (assuming it's class 1)
        if isinstance(y_pred, list):
            y_pred = y_pred[0]

        pred_prob_preictal = y_pred[:, 1]

        # Create the DataFrame
        results_df = pd.DataFrame({
            "pred_label": pred_labels,
            "true_label": true_labels,
            "pred_prob_preictal": pred_prob_preictal
        })

        filepath = os.path.join(self.config["model_path"], file_name)

        # Save to a text file
        results_df.to_csv(filepath, sep=';', index=False)
        print(f"Predictions saved to {file_name}")

    def _set_model_class(self):
        """
        Set the model class attribute based on the configuration.
        Returns:
            model_class (class): The model class based on the configuration.
        Raises:
            ValueError: If the model name in the configuration is not valid.
        """
        model_name = self.config["name"]
        print(f"Model name: {model_name}\n")
        if model_name == "MultiViewConvModel":
            self.model_class = MultiViewConvModel
        elif model_name == "MultiViewConvModel_v2":
            self.model_class = MultiViewConvModel_v2
        elif model_name == "MultiViewConvModelWithBatchNorm":
            self.model_class = MultiViewConvModelWithBatchNorm
        elif model_name == "MultiViewConvModelWithAttention":
            self.model_class = MultiViewConvModelWithAttention
        else:
            raise ValueError(f"Invalid model name '{model_name}' in the configuration.")