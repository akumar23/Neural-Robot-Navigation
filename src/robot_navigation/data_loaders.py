import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        from pathlib import Path
        # Get data file from data directory
        data_path = Path(__file__).parent.parent.parent / "data" / "training_data.csv"
        self.data = np.genfromtxt(data_path, delimiter=',')

        # Use StandardScaler instead of MinMaxScaler for better numerical stability
        # StandardScaler (z-score normalization) is more robust to outliers and
        # handles out-of-distribution values during inference better than MinMaxScaler
        # IMPORTANT: Only normalize features (all columns except the last one which is the label)
        self.scaler = StandardScaler()
        features = self.data[:, :-1]  # All columns except last (label)
        self.normalized_features = self.scaler.fit_transform(features)  # fits and transforms
        # Save scaler to models directory
        models_path = Path(__file__).parent.parent.parent / "models" / "scaler.pkl"
        pickle.dump(self.scaler, open(models_path, "wb"))  # save to normalize at inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # Use normalized features but ORIGINAL labels (0 or 1)
        # Labels should NOT be normalized for binary classification!
        n = self.normalized_features[idx]  # All features (already excludes label)
        y = self.data[idx, [-1]]  # Use original labels, not normalized

        x_tensor = torch.from_numpy(n).float()
        y_tensor = torch.from_numpy(y).float()
        dict1 = {}
        dict1 = {'input': x_tensor, 'label': y_tensor}
        return dict1


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size

        self.train_loader, self.test_loader = torch.utils.data.random_split(self.nav_dataset , [train_size, test_size])

class Sequence_Dataset(dataset.Dataset):
    """
    Dataset for LSTM training with variable-length sequences (episodes).

    Each episode is a sequence of timesteps containing:
    - features: 20D feature vector (already computed during collection)
    - collision: Binary collision label (0 or 1)

    Handles variable-length sequences and provides episode metadata.
    """
    def __init__(self, sequences_file='training_sequences.pkl'):
        from pathlib import Path

        # Load pickled sequences
        data_path = Path(__file__).parent.parent.parent / "data" / sequences_file
        with open(data_path, 'rb') as f:
            self.episodes = pickle.load(f)

        # Use same StandardScaler as feedforward model for consistency
        # For sequences, we'll fit the scaler on ALL timesteps from ALL episodes
        self.scaler = StandardScaler()

        # Collect all features from all episodes for scaler fitting
        all_features = []
        for episode in self.episodes:
            for timestep in episode:
                all_features.append(timestep['features'])

        all_features = np.array(all_features)
        self.scaler.fit(all_features)

        # Save scaler to models directory
        models_path = Path(__file__).parent.parent.parent / "models" / "scaler_lstm.pkl"
        pickle.dump(self.scaler, open(models_path, "wb"))

        print(f"Loaded {len(self.episodes)} episodes")
        total_timesteps = sum(len(ep) for ep in self.episodes)
        print(f"Total timesteps: {total_timesteps}")
        avg_length = total_timesteps / len(self.episodes)
        print(f"Average episode length: {avg_length:.1f}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        Get a single episode (sequence).

        Returns:
            Dictionary with:
            - input: Tensor of features (seq_len, 20)
            - label: Tensor of collision labels (seq_len,)
            - length: Actual sequence length (scalar)
        """
        if not isinstance(idx, int):
            idx = idx.item()

        episode = self.episodes[idx]
        seq_len = len(episode)

        # Extract features and labels
        features = np.array([timestep['features'] for timestep in episode])
        labels = np.array([timestep['collision'] for timestep in episode])

        # Normalize features
        features_normalized = self.scaler.transform(features)

        # Convert to tensors
        features_tensor = torch.from_numpy(features_normalized).float()
        labels_tensor = torch.from_numpy(labels).float()
        length_tensor = torch.tensor(seq_len, dtype=torch.long)

        return {
            'input': features_tensor,      # (seq_len, 20)
            'label': labels_tensor,        # (seq_len,)
            'length': length_tensor        # scalar
        }


def collate_sequences(batch):
    """
    Collate function for batching variable-length sequences.

    Pads sequences to the maximum length in the batch with zeros.
    Returns masks to indicate which timesteps are valid vs padding.

    Args:
        batch: List of episode dictionaries from Sequence_Dataset

    Returns:
        Dictionary with:
        - input: Padded sequences (batch, max_seq_len, features)
        - label: Padded labels (batch, max_seq_len)
        - length: Actual sequence lengths (batch,)
    """
    # Find max sequence length in this batch
    max_seq_len = max(item['length'].item() for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['input'].shape[1]  # Should be 20

    # Initialize padded tensors
    padded_inputs = torch.zeros(batch_size, max_seq_len, feature_dim)
    padded_labels = torch.zeros(batch_size, max_seq_len)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    # Fill in actual sequences
    for i, item in enumerate(batch):
        seq_len = item['length'].item()
        padded_inputs[i, :seq_len, :] = item['input']
        padded_labels[i, :seq_len] = item['label']
        lengths[i] = item['length']

    return {
        'input': padded_inputs,   # (batch, max_seq_len, 20)
        'label': padded_labels,   # (batch, max_seq_len)
        'length': lengths         # (batch,)
    }


class Sequence_Data_Loaders():
    """
    Data loaders for LSTM training with sequences.

    Splits episodes into train/test sets (80/20 split) and creates
    DataLoaders with proper batching and padding for variable-length sequences.
    """
    def __init__(self, batch_size, sequences_file='training_sequences.pkl'):
        self.sequence_dataset = Sequence_Dataset(sequences_file)

        # Split into train/test (80/20)
        train_size = int(0.8 * len(self.sequence_dataset))
        test_size = len(self.sequence_dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            self.sequence_dataset, [train_size, test_size]
        )

        # Create data loaders with custom collate function for padding
        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sequences
        )

        self.test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sequences
        )

        print(f"Train episodes: {train_size}, Test episodes: {test_size}")


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
