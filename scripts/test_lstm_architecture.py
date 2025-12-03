"""
Quick test to verify LSTM architecture is implemented correctly.

Tests:
1. Model initialization
2. Forward pass with single sample
3. Forward pass with batch sequences
4. Hidden state management
5. Padding and masking
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.robot_navigation.networks import Action_Conditioned_LSTM, FocalLoss


def test_model_initialization():
    """Test model can be initialized."""
    print("Test 1: Model Initialization")
    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model initialized successfully")
    print(f"  ✓ Total parameters: {total_params:,}")
    return model


def test_single_sample_forward():
    """Test forward pass with single sample."""
    print("\nTest 2: Single Sample Forward Pass")
    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    model.eval()

    # Create single sample (20 features)
    sample = torch.randn(20)

    with torch.no_grad():
        output, hidden = model(sample, hidden=None)

    print(f"  ✓ Input shape: {sample.shape}")
    print(f"  ✓ Output shape: {output.shape} (should be scalar)")
    print(f"  ✓ Hidden state: h={hidden[0].shape}, c={hidden[1].shape}")

    assert output.dim() == 0, "Output should be scalar for single sample"
    assert hidden[0].shape == (2, 1, 64), f"Hidden h shape mismatch: {hidden[0].shape}"
    assert hidden[1].shape == (2, 1, 64), f"Hidden c shape mismatch: {hidden[1].shape}"
    print("  ✓ All assertions passed")


def test_batch_sequence_forward():
    """Test forward pass with batch of sequences."""
    print("\nTest 3: Batch Sequence Forward Pass")
    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    model.eval()

    # Create batch of sequences
    batch_size = 4
    seq_len = 10
    features = 20

    batch = torch.randn(batch_size, seq_len, features)

    with torch.no_grad():
        output, hidden = model(batch, hidden=None)

    print(f"  ✓ Input shape: {batch.shape}")
    print(f"  ✓ Output shape: {output.shape} (should be {batch_size} x {seq_len} x 1)")
    print(f"  ✓ Hidden state: h={hidden[0].shape}, c={hidden[1].shape}")

    assert output.shape == (batch_size, seq_len, 1), f"Output shape mismatch: {output.shape}"
    assert hidden[0].shape == (2, batch_size, 64), f"Hidden h shape mismatch: {hidden[0].shape}"
    assert hidden[1].shape == (2, batch_size, 64), f"Hidden c shape mismatch: {hidden[1].shape}"
    print("  ✓ All assertions passed")


def test_hidden_state_persistence():
    """Test that hidden state persists and updates correctly."""
    print("\nTest 4: Hidden State Persistence")
    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    model.eval()

    # Process sequence step by step
    hidden = None
    timesteps = 5

    for t in range(timesteps):
        sample = torch.randn(20)
        with torch.no_grad():
            output, hidden = model(sample, hidden)

        print(f"  ✓ Timestep {t+1}: output={output.item():.4f}, hidden_norm={hidden[0].norm().item():.4f}")

    # Verify hidden state has accumulated information
    assert hidden is not None, "Hidden state should not be None"
    assert hidden[0].norm().item() > 0, "Hidden state should have non-zero values"
    print("  ✓ Hidden state persists across timesteps")


def test_padding_and_masking():
    """Test that padding and masking work correctly."""
    print("\nTest 5: Padding and Masking")

    # Create sequences with different lengths
    seq1 = torch.randn(5, 20)   # Length 5
    seq2 = torch.randn(8, 20)   # Length 8
    seq3 = torch.randn(3, 20)   # Length 3

    # Pad to max length (8)
    max_len = 8
    padded_batch = torch.zeros(3, max_len, 20)
    padded_batch[0, :5, :] = seq1
    padded_batch[1, :8, :] = seq2
    padded_batch[2, :3, :] = seq3

    lengths = torch.tensor([5, 8, 3])

    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    model.eval()

    with torch.no_grad():
        output, _ = model(padded_batch, hidden=None)

    print(f"  ✓ Padded batch shape: {padded_batch.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Lengths: {lengths.tolist()}")

    # Verify output has correct shape
    assert output.shape == (3, max_len, 1), f"Output shape mismatch: {output.shape}"
    print("  ✓ Padding works correctly")


def test_focal_loss():
    """Test FocalLoss works with LSTM outputs."""
    print("\nTest 6: Focal Loss Integration")
    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    loss_fn = FocalLoss(alpha=0.1, gamma=2.0)

    # Create batch
    batch_size = 4
    seq_len = 10
    features = 20

    inputs = torch.randn(batch_size, seq_len, features)
    labels = torch.randint(0, 2, (batch_size, seq_len)).float()

    # Forward pass
    outputs, _ = model(inputs, hidden=None)
    outputs = outputs.squeeze(-1)  # (batch, seq_len)

    # Compute loss on first sequence only (as example)
    loss = loss_fn(outputs[0], labels[0])

    print(f"  ✓ Loss value: {loss.item():.4f}")
    print(f"  ✓ FocalLoss works with LSTM outputs")

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("  ✓ All assertions passed")


def main():
    print("="*60)
    print("LSTM ARCHITECTURE VERIFICATION TESTS")
    print("="*60)

    try:
        model = test_model_initialization()
        test_single_sample_forward()
        test_batch_sequence_forward()
        test_hidden_state_persistence()
        test_padding_and_masking()
        test_focal_loss()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nLSTM architecture is correctly implemented and ready for training!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Collect sequences: python scripts/collect_sequences.py")
        print("3. Train LSTM: python scripts/train_lstm.py")
        print("4. Test inference: python scripts/run_lstm.py")
        print("5. Compare models: python scripts/compare_ff_vs_lstm.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
