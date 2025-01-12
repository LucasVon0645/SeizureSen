import os
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from keras.api.callbacks import History

from typing import Optional

def plot_training_history(history: History, save_dir: Optional[str] = None):

#def plot_training_history(history: History, save_dir: str | None = None):
    """
    Plots the training and validation loss over the epochs.

    Args:
        history (keras.callbacks.History): A Keras History object containing training and validation loss values.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))

    # Plot training & validation loss values
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)

    if save_dir:
        filepath = os.path.join(save_dir, "training_history.png")
        plt.savefig(filepath)


def save_confusion_matrix(y_true, y_pred, output_path):
    """
    Plot the confusion matrix and save it to a file.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_path (str): Path to save the plot.
    """

    # Generate the classification report
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Interictal", "Preictal"]
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.savefig(output_path)

def save_model_scores(classification_report_str, output_path):
    """
    Save the classification report to a file.

    Args:
        classification_report_str (str): The classification report as a string.
        output_path (str): Path to save the classification report.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(classification_report_str)
