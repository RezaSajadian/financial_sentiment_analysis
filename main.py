import logging
from data_processing import prepare_data
from model import get_model, get_trainer

# Configure logging
logging.basicConfig(filename='training.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def main():
    """Main function that runs the script."""
    # Log the start of the execution
    logging.info("Started execution")

    # Load the data from a text file
    with open('dataset.txt', 'r') as file:
        data = [line.strip().split(",") for line in file.readlines()]
        financial_texts = [(item[0], { '-1': 0, '0': 1, '1': 2 }[item[1]]) for item in data]

    
    # Use part of the data for testing (replace with actual test data in practice)
    test_data = financial_texts[:50]

    # Prepare the data
    logging.info("Preparing data...")
    train_dataset, test_dataset = prepare_data(financial_texts, test_data)

    # Get the model
    logging.info("Loading model...")
    model = get_model()

    # Get the trainer
    logging.info("Initializing trainer...")
    trainer = get_trainer(model, train_dataset)

    # Train the model
    logging.info("Starting training...")
    trainer.train()

    # Evaluate the model
    logging.info("Evaluating model...")
    eval_results = trainer.evaluate(test_dataset)

    # Log the evaluation results
    logging.info(f"Evaluation results: {eval_results}")

    # Log the end of the execution
    logging.info("Finished execution")

if __name__ == "__main__":
    main()
