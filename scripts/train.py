import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.data_loaders import Data_Loaders
from src.robot_navigation.networks import Action_Conditioned_FF, FocalLoss

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def calculate_collision_rate(data_loaders):
    """Calculate the collision rate from training data to set Focal Loss alpha."""
    total_samples = 0
    collision_samples = 0
    for sample in data_loaders.train_loader:
        labels = sample['label']
        total_samples += len(labels)
        collision_samples += (labels > 0.5).sum().item()
    return collision_samples / total_samples if total_samples > 0 else 0.1


def train_model(no_epochs):
    batch_size = 32
    data_loaders = Data_Loaders(batch_size)
    # Update model to use 20D input features (5 sensors + 6 spatial + 2 goal + 4 temporal + 2 spatial-goal + 1 action)
    model = Action_Conditioned_FF(input_size=20)

    # Calculate collision rate for Focal Loss alpha parameter
    collision_rate = calculate_collision_rate(data_loaders)
    print(f"Collision rate in training data: {collision_rate:.2%}")

    losses = []
    # Use Focal Loss to handle class imbalance (collisions are rare events)
    # alpha = collision_rate, gamma = 2.0 (recommended default)
    loss_function = FocalLoss(alpha=max(0.1, collision_rate), gamma=2.0)
    print(f"Using Focal Loss with alpha={max(0.1, collision_rate):.3f}, gamma=2.0")

    learning_rate = 0.01  # Initial learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduling: ReduceLROnPlateau reduces LR when validation loss plateaus
    # This allows high LR for fast initial convergence, then fine-tuning with lower LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # Reduce LR when loss stops decreasing
        factor=0.5,      # Reduce LR by half
        patience=10,     # Wait 10 epochs before reducing
        min_lr=1e-6      # Don't reduce below this
    )
    print(f"Using ReduceLROnPlateau scheduler with initial LR={learning_rate}")

    # Evaluate initial model
    model.eval()
    initial_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(initial_loss)
    print(f"Initial test loss: {initial_loss:.4f}")

    best_loss = float('inf')
    best_epoch = 0

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        # Use train_loader for training (not test_loader!)
        for idx, sample in enumerate(data_loaders.train_loader):
            optimizer.zero_grad()
            output = model(sample['input'])
            loss = loss_function(output, sample['label'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # Evaluate on test set
        model.eval()
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(test_loss)

        # Step the scheduler based on validation loss
        scheduler.step(test_loss)

        # Track best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch_i + 1
            # Save best model
            models_path = Path(__file__).parent.parent / "models" / "saved_model.pkl"
            torch.serialization.save(model.state_dict(), models_path)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch_i + 1) % 10 == 0 or epoch_i == 0:
            print(f"Epoch {epoch_i+1}/{no_epochs} - Train: {avg_train_loss:.4f}, "
                  f"Test: {test_loss:.4f}, LR: {current_lr:.6f}")

    print(f"\nTraining complete!")
    print(f"Best test loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"Final test loss: {test_loss:.4f}")
    return losses


if __name__ == '__main__':
    no_epochs = 100  # Train for more epochs for better accuracy
    start = time.time()
    train_model(no_epochs)
    end = time.time()
    print(f"Total training time: {end - start:.2f} seconds")
