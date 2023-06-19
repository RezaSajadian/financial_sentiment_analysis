

from transformers import BertForSequenceClassification, TrainingArguments, Trainer

def get_model():
    """Returns the BERT model for sequence classification."""
    return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def get_trainer(model, train_dataset):
    """Returns a Trainer instance."""
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
    )

    # Initialize the trainer
    return Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,
    )
