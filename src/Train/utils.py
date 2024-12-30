import os
import matplotlib.pyplot as plt

from keras.api.callbacks import History

def plot_training_history(history: History, save_dir: str|None = None):
    """
    Plots the training and validation loss over the epochs.

    Args:
        history (keras.callbacks.History): A Keras History object containing training and validation loss values.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))

    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    if save_dir:
        filepath = os.path.join(save_dir, "training_history.png")
        plt.savefig(filepath)
        
    plt.show()
    