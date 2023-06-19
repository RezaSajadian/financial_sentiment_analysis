# Import necessary libraries
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

# Instantiating the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def prepare_data(data):
    # This function prepares the data for training and testing.

    # Extract the text and corresponding labels from the data
    texts = [item[0] for item in data]

    # Convert labels to integers and handle potential issues
    labels = [int(item[1]) for item in data]
    labels = [max(label, 0) for label in labels]  # Ensuring there are no negative labels

    # Tokenize the text, add necessary tokens (like [CLS], [SEP]), limit the length of the sequences, and pad shorter sequences
    encodings = tokenizer(texts, truncation=True, padding=True)
    
    # Convert the encodings and labels into a PyTorch Dataset
    dataset = FinSentimentDataset(encodings, labels)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    return train_dataset, test_dataset


class FinSentimentDataset(Dataset):
    # This class will handle the conversion of data into PyTorch Dataset format.
    
    def __init__(self, encodings, labels):
        # Initialize the Dataset with the encodings and labels
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # This method returns an item from the dataset at the specified index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # This method returns the size of the dataset
        return len(self.labels)
