# Financial Sentiment Analysis with BERT

This project trains a BERT model for financial sentiment analysis.

## Prerequisites

- Python 3.7 or later
- PyTorch
- Transformers

## Installation

You can install the required packages using pip:

pip install -r requirements.txt


## Usage

To run the script, use the following command:

 `python main.py`

This will fine-tune a BERT model on the provided financial dataset and save the training logs in `training.log`.

## Files

- `dataset.py`: Contains the `FinancialDataset` class for handling the dataset.
- `data_processing.py`: Contains the `prepare_data` function for preparing the training and testing datasets.
- `model.py`: Contains the `get_model` and `get_trainer` functions for getting the model and the trainer.
- `main.py`: The main script that runs the whole process.

## Authors

- Reza Sajadian

## Acknowledgments

- This code uses the Transformers library by Hugging Face.
