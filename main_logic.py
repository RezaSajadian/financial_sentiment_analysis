# Import necessary libraries
import logging
from data_processing import prepare_data
from model import get_model, get_optimizer
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from config import get_config

# Load the configuration parameters from the config.ini file
config = get_config()

def main():
    # This is the main function that is called to run the financial sentiment analysis project.

    # Load the financial phrases dataset from the dataset.txt file
    with open('dataset.txt', 'r') as file:
        data = [line.strip().split(",") for line in file.readlines()]
        phrases = [(item[0], int(item[1])) for item in data]

    # Log the start of data preparation
    logging.info("Preparing data...")

    # Prepare the data for training and testing
    train_dataset, test_dataset = prepare_data(phrases)

    # Log the loading of the BERT model
    logging.info("Loading model...")

    # Load the BERT model
    model = get_model()

    # Log the setup of the training parameters
    logging.info("Setting up training...")

    # Compute the number of training steps and get the optimizer and scheduler
    num_training_steps = len(train_dataset) * int(config.get('TRAINING', 'EPOCHS'))
    optimizer, scheduler = get_optimizer(model, num_training_steps)

    # Define the training arguments for the BERT model
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=int(config.get('TRAINING', 'EPOCHS')),  # total number of training epochs
        per_device_train_batch_size=int(config.get('TRAINING', 'BATCH_SIZE')),  # batch size per device during training
        per_device_eval_batch_size=int(config.get('TRAINING', 'BATCH_SIZE')),  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    # Initialize the trainer with the BERT model and training arguments
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        optimizers=(optimizer, scheduler),  # optimizer and scheduler
    )

    # Log the start of training
    logging.info("Starting training...")

    # Train the model
    trainer.train()

    # Log the start of model evaluation
    logging.info("Evaluating model...")

    # Evaluate the model
    results = trainer.evaluate()

    # Log the results of model evaluation
    logging.info(f"Evaluation results: {results}")
