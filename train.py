import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from dataset import ASLDataset, asl_collate_fn
from model import ASLLSTM


"""
train.py
--------
Trains the ASLLSTM model on sequences extracted from MediaPipe keypoints.

The training loop:
- Loads training and validation sets
- Batches variable-length sequences via asl_collate_fn
- Trains using Adam optimizer
- Evaluates after each epoch
- Saves model to 'asl_model.pth'
"""


def train(
    json_path="Markup.json",
    num_classes=100,
    batch_size=4,
    lr=1e-3,
    epochs=30,
    hidden_dim=256
):

    # -----------------------------------------------------
    # Dataset & DataLoaders
    # -----------------------------------------------------
    train_ds = ASLDataset(json_path, split="train")
    val_ds   = ASLDataset(json_path, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=asl_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=asl_collate_fn
    )

    # -----------------------------------------------------
    # Model, optimizer, loss
    # -----------------------------------------------------
    model = ASLLSTM(
        input_dim=225,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=num_classes
    ).cpu()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------
    print("[INFO] Starting training...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0.0

        for X, y in train_loader:
            X = torch.tensor(X, dtype=torch.float32).cpu()
            y = torch.tensor(y, dtype=torch.long).cpu()

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping helps stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            pred = logits.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += y.size(0)
            running_loss += loss.item()

        train_acc = running_correct / running_total
        train_loss = running_loss / len(train_loader)

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = torch.tensor(X, dtype=torch.float32).cpu()
                y = torch.tensor(y, dtype=torch.long).cpu()

                logits = model(X)
                pred = logits.argmax(dim=1)

                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.3f} "
            f"- Val Acc: {val_acc:.3f}"
        )

        # Save checkpoint each epoch
        torch.save(model.state_dict(), "asl_model.pth")

    print("\n[DONE] Training complete. Model saved as 'asl_model.pth'.")


if __name__ == "__main__":
    train()