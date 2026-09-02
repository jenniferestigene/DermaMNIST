"""
train.py

"""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import DiseaseTypeCNN

image_size = 64


class DermaMNISTDataset(Dataset):
    """
    
    """

    def __init__(self, images_path: str, labels_path: str, train: bool = False):
        self.images = torch.from_numpy(np.load(images_path))
        self.labels = torch.from_numpy(np.load(labels_path))

        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """
    
    """

    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    weights = total / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    # Datasets and loaders
    train_dataset = DermaMNISTDataset(
        f"disease_train_images_{image_size}.npy", f"disease_train_labels_{image_size}.npy", train=True
    )
    val_dataset = DermaMNISTDataset(
        f"disease_val_images_{image_size}.npy", f"disease_val_labels_{image_size}.npy", train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # Model
    model = DiseaseTypeCNN(num_classes=7)
    model.to(device)

    # Weighted loss
    train_labels = np.load(f"disease_train_labels_{image_size}.npy")
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # LR Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # Logging
    log_rows = []
    num_epochs = 40
    best_val_loss = float("inf")

    # Early stopping
    early_stop_patience = 15
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs +1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation 
        model.eval()
        val_loss, val_correct, val_total = 0.00, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total 

        # Save a checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), "saved_model.pth")
            print(f" -> New best val_loss: {val_loss:.4f}, saved to saved_model.pth")
        else:
            epochs_since_improvement += 1

        # Step the scheduler
        scheduler.step(val_loss)

        # Read current LR for logging
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | "
              f"lr: {current_lr:.6f}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })


        # Early stopping check 
        if epochs_since_improvement >= early_stop_patience:
            print(f"\nNo val_loss improvement in {early_stop_patience} epochs. "
                  f"Stopping early at epoch {epoch}/{num_epochs}.")
            break


    # Write training log to CSV
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
                                )
        writer.writeheader()
        writer.writerows(log_rows)

    print("Saved training_log.csv")

    torch.save(model.state_dict(), "final_model.pth")
    print("Saved final_model.pth (last epoch, for reference/overfitting comparison)")


if __name__ == "__main__":
    train()