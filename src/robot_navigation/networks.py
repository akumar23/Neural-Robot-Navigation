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

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
