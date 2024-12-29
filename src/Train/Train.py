import json
import joblib
import os
import sys

from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Preprocessing.utils import load_preprocessed_data, transform_to_tensor
from src.Model.MultiViewConvModel import MultiViewConvModel


class ModelTrainer:
    def __init__(self, config_path, data_path):
        self.data_path = data_path
        self.config_path = config_path
        self.config = None
        self.model = None

        # Freq Domain Tensorflow Tensors (n_samples, n_channels, fft_bins, steps)
        self.X_train_freq = None
        self.X_test_freq = None

        # Time Domain Tensorflow Tensors (n_samples, n_channels, pca_bins, steps)
        self.X_train_time = None
        self.X_test_time = None

        # Label Tensors (n_samples,)
        self.y_train = None
        self.y_test = None

    def load_config(self):
        """
        Loads the configuration from a JSON file specified by self.config_path.
        The configuration is read from the file and stored in the self.config attribute
        as a dictionary.
        Dictionary keys are as follows:
        {
            "nb_filter": int,
            "l2": float,
            "dropout": float,
            "learning_rate": float,
            "model_time_steps": int,
            "nn_time_output": int,
	        "nn_freq_output": int,
            "batch_size": int,
            "nb_epoch": int,
	        "name": "first_test",
            "model_path": str
        }
        Obs.: model_path is the relative path from the project root.
        It is used to save/load the trained model
        Raises:
            FileNotFoundError: If the file specified by self.config_path does not exist.
            json.JSONDecodeError: If the file is not a valid JSON.
        """
        
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

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

    #! implement the train method
    def train(self):
        self._validate_input(train=True)
        
        X_fft = self.X_train_freq
        X_pca = self.X_train_time
        y_train = self.y_train
        config = self.config

        n_samples = X_fft.shape[0]
        channels = X_fft.shape[1]
        fft_bins = X_fft.shape[2]
        pca_bins = X_pca.shape[2]
        steps = X_fft.shape[3]

        X_fft = X_fft.reshape(n_samples, channels * fft_bins, steps, 1)
        X_pca = X_pca.reshape(n_samples, channels * pca_bins, steps, 1)

        model, early_stopping = MultiViewConvModel.get_model(
            config=config,
            channels=channels,
            fft_bins=fft_bins,
            pca_bins=pca_bins,
            steps=steps,
        )

        model.fit(
            {"time_domain_input": X_pca, "freq_domain_input": X_fft},
            {"final_output": y_train},
            epochs=config["nb_epoch"],
            verbose=1,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        #! finish the training process

    #! implement the evaluate method
    def evaluate(self):
        self._validate_input(train=False)
        pass

    def save_model(self):
        """
        Saves the trained model to the specified file path.

        The file path is retrieved from the configuration dictionary (`self.config["model_path"]`).
        The model is serialized and saved using the `joblib` library.

        Raises:
            KeyError: If "model_path" is not found in the configuration dictionary.
            Exception: If there is an error during the model saving process.
        """
        model_path = self.config["model_path"]
        joblib.dump(self.model, model_path)

    def load_model(self):
        """
        Loads a pre-trained model from the specified file path in the configuration.

        This method uses the joblib library to load the model from the file path
        provided in the configuration dictionary under the key "model_path". The
        loaded model is then assigned to the instance variable `self.model`.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            Exception: If there is an error during the loading of the model.
        """
        model_path = self.config["model_path"]
        self.model = joblib.load(model_path)

    def _validate_input(self, train=True):
        """
        Validates the input data for training or testing.
        Parameters:
        train (bool): If True, validates the training data. If False, validates the testing data.
        Raises:
        ValueError: If the number of samples in X_fft_train, X_pca_train, and y_train are not equal (when train=True).
                    If the number of time steps in X_fft_train and X_pca_train are not equal (when train=True).
                    If the number of samples in X_fft_test, X_pca_test, and y_test are not equal (when train=False).
                    If the number of time steps in X_fft_test and X_pca_test are not equal (when train=False).
        """
        if train:
            X_fft_train = self.X_train_freq
            X_pca_train = self.X_train_time
            y_train = self.y_train

            if X_fft_train.shape(0) != X_pca_train.shape(0) or X_fft_train.shape(
                0
            ) != y_train.shape(0):
                raise ValueError(
                    "Number of samples in X_fft_train, X_pca_train, and y_train must be equal."
                )
            if X_fft_train.shape(-1) != X_pca_train.shape(-1):
                raise ValueError(
                    "Number of time steps in X_fft_train, X_pca_train must be equal."
                )

        else:
            X_fft_test = self.X_train_freq
            X_pca_test = self.X_test_time
            y_test = self.y_test

            if X_fft_test.shape(0) != X_pca_test.shape(0) or X_fft_test.shape(
                0
            ) != y_test.shape(0):
                raise ValueError(
                    "Number of samples in X_fft_test, X_pca_test, and y_test must be equal."
                )
            if X_fft_test.shape(-1) != X_pca_test.shape(-1):
                raise ValueError(
                    "Number of time steps in X_fft_test, X_pca_test must be equal."
                )


if __name__ == "__main__":
    config_path = Path("models", "config", "first_test_cfg.json")
    data_path = os.path.join("data", "preprocessed", "Dog_1")

    trainer = ModelTrainer(config_path, data_path)
    trainer.load_config()
    trainer.load_data()
    # trainer.train()
    # trainer.evaluate()
    # trainer.save_model()
