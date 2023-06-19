from transformers import BertTokenizer
from dataset import FinancialDataset

def prepare_data(financial_texts, test_data):
    """Prepares the data for training the model."""

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_texts, train_labels = zip(*financial_texts)
    test_texts, test_labels = zip(*test_data)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

    # Prepare Dataset
    train_dataset = FinancialDataset(train_encodings, train_labels)
    test_dataset = FinancialDataset(test_encodings, test_labels)

    return train_dataset, test_dataset
