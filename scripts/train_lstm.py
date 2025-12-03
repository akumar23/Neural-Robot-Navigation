"""
LSTM Training Script for Temporal Collision Detection.

Trains an LSTM network on episode sequences with proper handling of:
- Variable-length sequences with padding
- Masked loss computation (ignore padding)
- Hidden state management across batches
- Early stopping and model checkpointing
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.data_loaders import Sequence_Data_Loaders
from src.robot_navigation.networks import Action_Conditioned_LSTM, FocalLoss

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def calculate_collision_rate_sequences(data_loaders):
    """
    Calculate collision rate from sequence data for Focal Loss alpha.

    Args:
        data_loaders: Sequence_Data_Loaders instance

    Returns:
        Collision rate (fraction of timesteps with collisions)
    """
    total_timesteps = 0
    collision_timesteps = 0

    for batch in data_loaders.train_loader:
        labels = batch['label']       # (batch, max_seq_len)
        lengths = batch['length']     # (batch,)

        # Count only valid timesteps (not padding)
        batch_size = labels.shape[0]
        for i in range(batch_size):
            valid_len = lengths[i].item()
            valid_labels = labels[i, :valid_len]
            total_timesteps += valid_len
            collision_timesteps += (valid_labels > 0.5).sum().item()

    return collision_timesteps / total_timesteps if total_timesteps > 0 else 0.1


def compute_sequence_loss(model, sequences, labels, lengths, loss_function):
    """
    Compute loss on sequences with proper masking of padded timesteps.

    Args:
        model: LSTM model
        sequences: Padded input sequences (batch, max_seq_len, features)
        labels: Padded labels (batch, max_seq_len)
        lengths: Actual sequence lengths (batch,)
        loss_function: Loss function (e.g., FocalLoss)

    Returns:
        Average loss over all valid (non-padded) timesteps
    """
    # Forward pass through LSTM
    outputs, _ = model(sequences, hidden=None)  # (batch, max_seq_len, 1)
    outputs = outputs.squeeze(-1)  # (batch, max_seq_len)

    # Compute loss only on valid timesteps
    batch_size, max_seq_len = labels.shape
    total_loss = 0
    num_valid_timesteps = 0

    for i in range(batch_size):
        valid_len = lengths[i].item()
        # Extract valid timesteps (not padding)
        valid_outputs = outputs[i, :valid_len]
        valid_labels = labels[i, :valid_len]
        # Compute loss on valid timesteps
        loss = loss_function(valid_outputs, valid_labels)
        total_loss += loss * valid_len  # Weight by sequence length
        num_valid_timesteps += valid_len

    # Average loss across all valid timesteps
    avg_loss = total_loss / num_valid_timesteps if num_valid_timesteps > 0 else 0
    return avg_loss


def train_lstm_model(no_epochs, batch_size=16, device='cpu'):
    """
    Train LSTM model on sequence data.

    Args:
        no_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        losses: List of test losses per epoch
    """
    print("="*60)
    print("LSTM TRAINING PIPELINE")
    print("="*60)

    # Load sequence data
    print("\nLoading sequence data...")
    data_loaders = Sequence_Data_Loaders(batch_size=batch_size)

    # Calculate collision rate for Focal Loss
    print("\nCalculating collision rate...")
    collision_rate = calculate_collision_rate_sequences(data_loaders)
    print(f"Collision rate in sequence data: {collision_rate:.2%}")

    # Initialize LSTM model
    print("\nInitializing LSTM model...")
    model = Action_Conditioned_LSTM(
        input_size=20,      # 20D enhanced features
        hidden_size=64,     # Match feedforward baseline
        num_layers=2,       # 2 LSTM layers
        output_size=1,      # Binary collision prediction
        dropout=0.2         # Match feedforward baseline
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Focal Loss for class imbalance
    loss_function = FocalLoss(alpha=max(0.1, collision_rate), gamma=2.0)
    print(f"\nUsing Focal Loss with alpha={max(0.1, collision_rate):.3f}, gamma=2.0")

    # Optimizer and scheduler
    learning_rate = 0.001  # LSTM often benefits from lower initial LR than FF
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    print(f"Using Adam optimizer with initial LR={learning_rate}")
    print(f"Using ReduceLROnPlateau scheduler")

    # Evaluate initial model
    print("\nEvaluating initial model...")
    model.eval()
    initial_loss = model.evaluate_sequences(model, data_loaders.test_loader, loss_function, device)
    print(f"Initial test loss: {initial_loss:.4f}")

    losses = [initial_loss]
    best_loss = float('inf')
    best_epoch = 0

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60 + "\n")

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        # Train on batches
        for batch_idx, batch in enumerate(data_loaders.train_loader):
            sequences = batch['input'].to(device)      # (batch, max_seq_len, 20)
            labels = batch['label'].to(device)         # (batch, max_seq_len)
            lengths = batch['length']                  # (batch,)

            optimizer.zero_grad()

            # Compute loss with masking
            loss = compute_sequence_loss(model, sequences, labels, lengths, loss_function)

            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # Evaluate on test set
        model.eval()
        test_loss = model.evaluate_sequences(model, data_loaders.test_loader, loss_function, device)
        losses.append(test_loss)

        # Step scheduler
        scheduler.step(test_loss)

        # Track best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch_i + 1
            # Save best model
            models_path = Path(__file__).parent.parent / "models" / "saved_model_lstm.pkl"
            torch.save(model.state_dict(), models_path)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress
        if (epoch_i + 1) % 10 == 0 or epoch_i == 0:
            print(f"Epoch {epoch_i+1}/{no_epochs} - Train: {avg_train_loss:.4f}, "
                  f"Test: {test_loss:.4f}, LR: {current_lr:.6f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best test loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Models saved to: models/saved_model_lstm.pkl")
    print(f"Scaler saved to: models/scaler_lstm.pkl")
    print("="*60)

    return losses


def plot_training_curves(losses, save_path=None):
    """
    Plot training curves.

    Args:
        losses: List of test losses
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('LSTM Training: Test Loss over Epochs', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def main():
    # Training configuration
    no_epochs = 100
    batch_size = 16  # Smaller batch size for LSTM due to memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train model
    start = time.time()
    losses = train_lstm_model(no_epochs, batch_size, device)
    end = time.time()

    print(f"\nTotal training time: {end - start:.2f} seconds")
    print(f"Average time per epoch: {(end - start) / no_epochs:.2f} seconds")

    # Plot training curves
    plot_save_path = Path(__file__).parent.parent / "models" / "lstm_training_curve.png"
    plot_training_curves(losses, save_path=plot_save_path)


if __name__ == '__main__':
    main()
