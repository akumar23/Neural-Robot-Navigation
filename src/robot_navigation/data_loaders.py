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

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
