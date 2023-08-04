from src.model import DialogueActSlotValueModel
from src.utils.constant import MAX_TOKENS, DATA_PATH, SLOT_TO_ID, ID_TO_SLOT

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score

class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['utterance'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        targets = self.tokenizer(item['value_labels'], padding='max_length', truncation=True, max_length=32, return_tensors='pt').input_ids
        return inputs.input_ids.squeeze(), targets.squeeze()

def train():
    with open('data/train_value.json') as f:
        train_data = json.load(f)
    with open('data/test_value.json') as f:
        test_data = json.load(f)
    
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    train_dataset = MyDataset(train_data, tokenizer)
    test_dataset = MyDataset(test_data, tokenizer)

    
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'utterance', 'act_str'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create an optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    # Train the model
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Adjust batch size to suit your GPU memory
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)  

    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*2, num_training_steps=len(train_dataloader)*10)

    # Training loop
    model.train()  # Set the model to training mode
    for epoch in range(10):  # Number of epochs
        total_loss = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, attention_mask=(inputs!=tokenizer.pad_token_id).long(), labels=targets)

            # Loss
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()

            # Update learning rate
            scheduler.step()

        print(f"Epoch: {epoch}, Loss: {total_loss / len(train_data)}")

        model.eval()  # Set model to evaluation mode
        total_valid_loss = 0
        with torch.no_grad():  # Do not calculate gradients for validation, saves memory and computations
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs, attention_mask=(inputs!=tokenizer.pad_token_id).long(), labels=targets)

                # Loss
                loss = outputs.loss
                total_valid_loss += loss.item()

        print(f"Validation Epoch: {epoch}, Loss: {total_valid_loss / len(test_data)}")

        model.train()  # Set the model back to training mode

    # Switch model back to evaluation mode
    model.eval()

