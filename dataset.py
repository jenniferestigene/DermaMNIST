"""
dataset.py


"""


import numpy as np
import matplotlib.pyplot as plt 
from medmnist import DermaMNIST


# Class order, verified against MedMNIST's own label mapping
CLASS_NAMES = [
    "actinic keratoses and intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]


image_size = 128


def load_dermamnist(split: str):
    """
    Load one split of DermaMNIST and convert it to a PyTorch-convention
    Numpy array.

    Args:
        split: one of "train", "val", "test".

    Returns:
        images: np.ndarray, shape (N, 3, image_size, image_size), float32, values in [0,1]
        labels: np.ndarray, shape (N,), int64 class indices (0-6)
    """

    # download=True caches the .npz locally, so this only fetches from the network once, not on every run
    dataset = DermaMNIST(split=split, download=True, size=image_size)

    images = dataset.imgs
    labels = dataset.labels.squeeze()

    # Normalize pixel values from [0, 255] to [0, 1]
    # Reorder axes from channel-last to channel-first
    images = images.astype(np.float32) / 255.0
    images = images.transpose(0, 3, 1, 2)
    labels = labels.astype(np.int64)


    return images, labels 


def save_class_distribution_plot(labels: np.ndarray, out_path:str):
    """
    Building a bar chart of class frequency in the training set
    and saving it to out_path.

    Args: 
        labels: np.ndarray of integer class indices
        out_path: file path to save the figure to, e.g. "assets/class_distribution.png"
    """

    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    print(dict(zip(CLASS_NAMES, counts)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(CLASS_NAMES, counts, color="#4C72B0")
    ax.set_xlabel("Disease type")
    ax.set_ylabel("Number of images")
    ax.set_title("DermaMNIST class distribution (training set)")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved class distribution plot to {out_path} "
          f"(min: {counts.min()}, max: {counts.max()}, "
          f"imbalance ratio: {counts.max() / counts.min():.2f}x)")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        images, labels = load_dermamnist(split)
        print(f"[{split}] images: {images.shape}, dtype: {images.dtype} | "
              f"labels: {labels.shape}, dtype: {labels.dtype}")

        np.save(f"disease_{split}_images_{image_size}.npy", images)
        np.save(f"disease_{split}_labels_{image_size}.npy", labels)


    train_labels = np.load(f"disease_train_labels_{image_size}.npy")
    save_class_distribution_plot(train_labels, "assets/class_distribution.png")
