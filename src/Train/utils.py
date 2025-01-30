import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

from keras.api.callbacks import History

from typing import Optional

def plot_training_history(history: History, save_dir: Optional[str] = None, suffix: str = ""):

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
        filename = f"training_history{suffix}.png"
        filepath = os.path.join(save_dir, filename)
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

def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, model_path, filename = "roc_curve.png"):
    """
    Generate and save the ROC curve plot.

    Parameters:
        y_true_classes (np.ndarray): The true class labels.
        y_pred_prob (np.ndarray): The predicted probabilities (probabilities for class 1).
        filename (str): The filename of the ROC curve plot.
        model_path (str): The path where the plot will be saved.
    """
    if isinstance(y_pred, list):
        y_pred_prob = y_pred[0][:, 1]
    elif y_pred.ndim == 2:
        y_pred_prob = y_pred[:, 1]
    else:
        y_pred_prob = y_pred
        
    if y_true.ndim == 2:
        y_true_classes = y_true[:, 1]
    else:
        y_true_classes = y_true
        
    
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(y_true_classes, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Save the ROC curve plot
    plt.savefig(os.path.join(model_path, filename))
    plt.close()

    print(f"ROC curve saved to {os.path.join(model_path, filename)}")

def print_and_save_cross_validation_results(all_metrics: dict, model_path: str, filename="cross_validation_results.txt"):
    """
    Print the average metrics and the standard deviations from cross-validation.

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

    std_metrics = {
        "accuracy": np.std([metrics["accuracy"] for metrics in all_metrics]),
        "interictal_precision": np.std(
        [metrics["interictal"]["precision"] for metrics in all_metrics]
        ),
        "interictal_recall": np.std(
        [metrics["interictal"]["recall"] for metrics in all_metrics]
        ),
        "interictal_f1-score": np.std(
        [metrics["interictal"]["f1-score"] for metrics in all_metrics]
        ),
        "preictal_precision": np.std(
        [metrics["preictal"]["precision"] for metrics in all_metrics]
        ),
        "preictal_recall": np.std(
        [metrics["preictal"]["recall"] for metrics in all_metrics]
        ),
        "preictal_f1-score": np.std(
        [metrics["preictal"]["f1-score"] for metrics in all_metrics]
        ),
    }

    print("\n\nCross-Validation Results:")
    with open(os.path.join(model_path, filename), "w", encoding="utf-8") as f:
        for key, description in metrics_to_print.items():
            print(f"{description}: {avg_metrics[key]:.4f} +/- {std_metrics[key]:.4f}")
            f.write(f"{description}: {avg_metrics[key]:.4f} +/- {std_metrics[key]:.4f}\n")
