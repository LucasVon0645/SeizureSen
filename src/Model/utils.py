import os
import sys
import joblib
import json
from typing import Optional
from keras.api.models import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def load_model_from_config(config: dict, class_model) -> Model:
    """
    Load a pre-trained model from a configuration dictionary.
    Args:
        config (dict): A dictionary containing configuration parameters.
                       It must include the key "model_path" which specifies
                       the directory where the model checkpoint is stored.
        class_model (class): The class of the model to be loaded.
    Returns:
        Model: The loaded model with weights restored from the checkpoint.
    Raises:
        KeyError: If the "model_path" key is not found in the config dictionary.
        OSError: If there is an issue loading the model weights from the specified filepath.
    """

    model_path = config["model_path"]
    filename = "best_model.keras"
    filepath = os.path.join(model_path, "checkpoint", filename)

    model, _ = class_model.get_model(config)

    model.load_weights(filepath)

    print("\n\nModel loaded successfully from ", filepath)

    return model


def load_scalers_from_config(config: dict) -> tuple[list[dict], list[dict]]:
    """
    Load time domain and frequency domain scalers from a configuration dictionary.
    This function reads a configuration dictionary to determine the path and filename
    of a pickle file containing the scalers. It then loads the scalers from the file
    and returns the time domain and frequency domain scalers.
    Args:
        config (dict): A dictionary containing the configuration. It must have the keys:
            - "model_path" (str): The path to the directory containing the scaler file.
            - "name" (str): The base name of the model.
    Returns:
        tuple: A tuple containing two elements:
            - scalers_time: The time domain scalers.
            - scalers_freq: The frequency domain scalers.
    """

    model_path = config["model_path"]
    filename = "feature_scalers.pkl"

    filepath = os.path.join(model_path, filename)

    # Load the scalers from the file
    scalers = joblib.load(filepath)

    # Access the time domain and frequency domain scalers
    scalers_time = scalers["time_domain"]
    scalers_freq = scalers["frequency_domain"]
    
    print("Scalers loaded successfully from ", filepath)
    
    return scalers_time, scalers_freq

def load_config(config_path) -> Optional[dict]:

    #def load_config(config_path) -> dict | None:
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
        "nn_time_output": int,
        "nn_freq_output": int,
        "batch_size": int,
        "nb_epoch": int,
        "name": str,
        "model_path": str,
        "model_time_steps": int,
        "channels": int,
        "fft_bins": int,
        "pca_bins": int,
        "preictal_class_weight": float,
        "use_early_exits": bool,
        "augmentation_strategy": "SMOTE" | "ADASYN" | null,
        
    }
    Obs.: model_path is the relative path from the project root.
    It is used to save/load the trained model, as well the scalers.
    Raises:
        FileNotFoundError: If the file specified by self.config_path does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    config = None
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config