# Import necessary libraries
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

def get_model():
    # This function loads the pre-trained BERT model and prepares it for sequence classification.
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    return model

def get_optimizer(model, steps):
    # This function prepares the optimizer and the learning rate scheduler for training the model.
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    return optimizer, scheduler
