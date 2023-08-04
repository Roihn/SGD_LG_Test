from src.model import DialogueActSlotValueModel
from src.utils.constant import MAX_TOKENS, DATA_PATH, SLOT_TO_ID, ID_TO_SLOT

import json
import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, TrainingArguments, Trainer
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
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    train_dataset = load_dataset('json', data_files='data/train_value.json', split='train')
    val_dataset = load_dataset('json', data_files='data/test_value.json', split='train')
    
    def preprocess_data(examples):
        inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        targets = tokenizer(examples['value_labels'], padding='max_length', truncation=True, max_length=32, return_tensors='pt').input_ids
        inputs['labels'] = targets
        return inputs
    
    train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    # train_dataset.set_format('torch')
    val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names)
    # val_dataset.set_format('torch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_size = 4
    metrics = "f1"
    training_args = TrainingArguments(
        output_dir='./output',
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=40, 
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=2e-5,
        metric_for_best_model=metrics,
        evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    
    trainer.save_model("model_value")