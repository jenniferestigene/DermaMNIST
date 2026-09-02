"""
evaluate.py

Evaluates the best DermaMNIST checkpoint (saved_model.pth) on the
held-out test set. Reports overall accuracy plus per-class precision,
recall, and F1 so classes underrepresented in training (see class
imbalance ratio in dataset.py output) can be checked individually,
not masked by overall accuracy.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from model import DiseaseTypeCNN
from train import DermaMNISTDataset

image_size = 128

CLASS_NAMES = [
    "actinic keratoses and intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]


def load_model(checkpoint_path: str, device: torch.device) -> DiseaseTypeCNN:
    """
    Load DiseaseTypeCNN with weights from a saved state_dict checkpoint.

    Args:
        checkpoint_path: path to a .pth file saved via model.state_dict()
        device: torch.device to load the model onto

    Returns:
        model in eval mode, on device
    """
    model = DiseaseTypeCNN(num_classes=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def run_inference(model: DiseaseTypeCNN, loader: DataLoader, device: torch.device):
    """
    Run the model over a DataLoader and collect predictions and true labels.

    Args:
        model: trained model in eval mode
        loader: DataLoader for the split to evaluate
        device: torch.device to run inference on

    Returns:
        y_true: np.ndarray of true class indices
        y_pred: np.ndarray of predicted class indices
    """
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)

    return y_true, y_pred


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """
    Compute and save a confusion matrix heatmap for the test set predictions.

    Args:
        y_true: np.ndarray of true class indices
        y_pred: np.ndarray of predicted class indices
        out_path: file path to save the figure to, e.g. "assets/confusion_matrix.png"
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("DermaMNIST confusion matrix (test set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix to {out_path}")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = DermaMNISTDataset(
        f"disease_test_images_{image_size}.npy", f"disease_test_labels_{image_size}.npy", train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Best checkpoint only, per convention (not final_model.pth)
    model = load_model("saved_model.pth", device)

    y_true, y_pred = run_inference(model, test_loader, device)

    overall_acc = (y_true == y_pred).mean()
    print(f"\nTest accuracy: {overall_acc:.4f}\n")

    print("Per-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    save_confusion_matrix_plot(y_true, y_pred, "assets/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()