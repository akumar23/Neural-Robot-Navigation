import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in collision detection.

    Focal Loss down-weights easy examples (confident predictions) and focuses
    learning on hard examples near the decision boundary. This is critical for
    collision detection where collisions are rare events (<10% of samples).

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the positive class (collisions).
               Should be set to collision_rate (e.g., 0.1 for 10% collisions).
        gamma: Focusing parameter. Higher values focus more on hard examples.
               Recommended: 2.0 (default in RetinaNet paper).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Get probabilities from logits
        p = torch.sigmoid(inputs)

        # Clamp for numerical stability
        p = torch.clamp(p, min=1e-7, max=1-1e-7)

        # Compute pt (probability of the correct class)
        pt = p * targets + (1 - p) * (1 - targets)

        # Compute alpha_t (alpha for positive class, 1-alpha for negative)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross entropy: -log(pt)
        ce = -torch.log(pt)

        # Focal loss
        F_loss = alpha_t * focal_weight * ce

        return F_loss.mean()


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper networks.

    Skip connections enable training significantly deeper networks by addressing
    the vanishing gradient problem. The output is: ReLU(x + F(x)) where F(x)
    is the learned residual transformation.

    Uses LayerNorm instead of BatchNorm for compatibility with single samples.
    """
    def __init__(self, hidden_size, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # Save input for skip connection

        out = self.linear1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.ln2(out)

        out = out + identity  # Skip connection
        out = self.relu(out)

        return out


class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=1, num_res_blocks=3):
        """
        Action-Conditioned Feedforward Network with Residual Connections.

        Deeper network with skip connections for better gradient flow and
        increased capacity to learn complex collision boundaries.

        Args:
            input_size: Number of input features (default 12 with feature engineering):
                        5 raw sensors + 6 derived features + 1 action
            hidden_size: Size of hidden layers (increased to 64 for more capacity)
            output_size: Output size (1 for binary collision prediction)
            num_res_blocks: Number of residual blocks (default 3)
        """
        super(Action_Conditioned_FF, self).__init__()

        # Input projection to hidden dimension
        # Uses LayerNorm for compatibility with single samples during inference
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate=0.2)
            for _ in range(num_res_blocks)
        ])

        # Output head with gradual dimensionality reduction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        # Note: No sigmoid on output - Focal Loss includes sigmoid via BCE

    def forward(self, x):
        # Handle both single samples and batches
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)  # Add batch dimension

        # Input projection
        out = self.input_projection(x)

        # Pass through residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)

        # Output head
        out = self.output_head(out)

        if single_sample:
            out = out.squeeze(0)  # Remove batch dimension

        return out

    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0
        num_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                output = self.forward(sample['input'])
                loss = loss_function(output, sample['label'])
                total_loss += loss.item()
                num_samples += 1
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss

class Action_Conditioned_LSTM(nn.Module):
    """
    Action-Conditioned LSTM Network with Temporal Memory for Collision Detection.

    Captures temporal dependencies in robot navigation by maintaining hidden state
    across timesteps. This enables the model to learn temporal patterns like:
    - Consequence of previous actions on current state
    - Oscillation/thrashing detection
    - Momentum and trajectory understanding
    - Context-aware decision making based on action history

    Architecture Design Decisions:
    - LSTM vs GRU: Using LSTM for better long-term memory capability
      - LSTM has separate cell state and hidden state for information flow
      - Better at capturing long-range dependencies (important for escape behaviors)
      - Slightly slower but more powerful than GRU
    - Layers: 2 LSTM layers for hierarchical temporal feature learning
      - Layer 1: Low-level temporal patterns (recent action effects)
      - Layer 2: High-level temporal patterns (navigation strategies)
    - Hidden size: 64 units (matching feedforward baseline for fair comparison)
    - Dropout: 0.2 between LSTM layers and before output head

    Args:
        input_size: Number of input features (20 with enhanced features)
        hidden_size: LSTM hidden state dimension (default: 64)
        num_layers: Number of stacked LSTM layers (default: 2)
        output_size: Output dimension (1 for binary collision prediction)
        dropout: Dropout rate between LSTM layers (default: 0.2)
    """
    def __init__(self, input_size=20, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(Action_Conditioned_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers with dropout between layers
        # batch_first=True: input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout only if multiple layers
        )

        # Output head: LSTM hidden state -> collision logit
        # Use gradual dimensionality reduction for better feature extraction
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),  # Normalize LSTM output
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        # Note: No sigmoid on output - FocalLoss includes sigmoid

    def forward(self, x, hidden=None):
        """
        Forward pass with optional hidden state.

        Supports both batch sequences (training) and single-step inference:
        - Training: x is (batch, seq_len, features), processes full sequences
        - Inference: x is (batch, 1, features) or (1, features), processes single timestep

        Args:
            x: Input tensor
               Training: (batch, seq_len, features)
               Inference: (batch, 1, features) or (features,) or (1, features)
            hidden: Tuple of (h_0, c_0) hidden states, or None
                   h_0: (num_layers, batch, hidden_size) - hidden state
                   c_0: (num_layers, batch, hidden_size) - cell state
                   If None, initialized to zeros

        Returns:
            output: Collision logits
                   Training: (batch, seq_len, 1)
                   Inference: (batch, 1) or scalar
            hidden: Tuple of (h_n, c_n) updated hidden states
        """
        # Handle different input shapes for inference vs training
        single_sample = False
        single_timestep = False

        if x.dim() == 1:
            # Shape: (features,) -> add batch and sequence dimensions
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, features)
            single_sample = True
            single_timestep = True
        elif x.dim() == 2:
            # Could be (batch, features) or (1, features)
            if x.size(0) == 1 or x.size(1) == self.input_size:
                # Add sequence dimension: (batch, features) -> (batch, 1, features)
                x = x.unsqueeze(1)
                single_timestep = True
        # else: x.dim() == 3, already (batch, seq_len, features)

        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size)
        # hidden: tuple of (h_n, c_n), each (num_layers, batch, hidden_size)
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply output head to all timesteps
        # Reshape to (batch * seq_len, hidden_size) for linear layers
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, self.hidden_size)

        # Pass through output head
        output = self.output_head(lstm_out_flat)

        # Reshape back to (batch, seq_len, output_size)
        output = output.reshape(batch_size, seq_len, self.output_size)

        # Remove dimensions for single sample/timestep inference
        if single_sample and single_timestep:
            output = output.squeeze(0).squeeze(0)  # Scalar
        elif single_timestep:
            output = output.squeeze(1)  # (batch, output_size)

        return output, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state to zeros.

        Args:
            batch_size: Batch size for hidden state
            device: Device to create tensors on ('cpu' or 'cuda')

        Returns:
            Tuple of (h_0, c_0), each (num_layers, batch, hidden_size)
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)

    def evaluate_sequences(self, model, test_loader, loss_function, device='cpu'):
        """
        Evaluate model on sequence data with proper masking of padded timesteps.

        Args:
            model: LSTM model
            test_loader: DataLoader with sequences
            loss_function: Loss function (e.g., FocalLoss)
            device: Device to run on

        Returns:
            Average loss over all valid (non-padded) timesteps
        """
        model.eval()
        total_loss = 0
        num_valid_timesteps = 0

        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['input'].to(device)  # (batch, max_seq_len, features)
                labels = batch['label'].to(device)      # (batch, max_seq_len)
                lengths = batch['length']               # (batch,) - actual sequence lengths

                # Forward pass
                outputs, _ = model(sequences, hidden=None)  # (batch, max_seq_len, 1)
                outputs = outputs.squeeze(-1)  # (batch, max_seq_len)

                # Compute loss only on valid timesteps (not padding)
                batch_size, max_seq_len = labels.shape
                for i in range(batch_size):
                    valid_len = lengths[i].item()
                    # Loss on valid timesteps only
                    valid_outputs = outputs[i, :valid_len]
                    valid_labels = labels[i, :valid_len]
                    loss = loss_function(valid_outputs, valid_labels)
                    total_loss += loss.item() * valid_len
                    num_valid_timesteps += valid_len

        avg_loss = total_loss / num_valid_timesteps if num_valid_timesteps > 0 else 0
        return avg_loss


def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
