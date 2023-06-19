from torch.utils.data import Dataset
import torch

class FinancialDataset(Dataset):
    def __init__(self, encodings, labels):
        """Initializes the dataset with encodings and labels."""
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """Returns a single encoded item."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.labels)
