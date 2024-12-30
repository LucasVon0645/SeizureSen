import os
import joblib
import json
from keras.api.models import load_model, Model

def load_model_from_config(config: dict) -> Model:
    """
    Loads a pre-trained model from the specified file path in the configuration.

    This method uses the keras library to load the model from the file path
    provided in the configuration dictionary under the key "model_path" and
    "name". The loaded model is then assigned to the instance variable `self.model`.
    File format: ".keras"
    Parameters:
        config (dict): A dictionary containing the configuration settings
                       for the model.
    Raises:
        FileNotFoundError: If the specified model file does not exist.
        Exception: If there is an error during the loading of the model.
    """
    model_path = config["model_path"]
    model_name = config["name"]
    filename = model_name + ".keras"

    filepath = os.path.join(model_path, filename)

    model = load_model(filepath, compile=True)

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
    filename = config["name"] + "_scalers.pkl"

    filepath = os.path.join(model_path, filename)

    # Load the scalers from the file
    scalers = joblib.load(filepath)

    # Access the time domain and frequency domain scalers
    scalers_time = scalers['time_domain_scalers']
    scalers_freq = scalers['frequency_domain_scalers']

    return scalers_time, scalers_freq

def load_config(config_path) -> dict | None:
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
    It is used to save/load the trained model, as well the scalers.
    Raises:
        FileNotFoundError: If the file specified by self.config_path does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    config = None
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
        
    return config