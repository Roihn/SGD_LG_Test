from src.model import DialogueActSlotValueModel
from src.utils.constant import MAX_TOKENS, DATA_PATH, ACT_TO_ID, SLOT_TO_ID, ID_TO_ACT, ID_TO_SLOT

import json
import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch import LongTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

class SGDDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        example = self.data[index]

        # Convert data to tensors
        for key in example:
            example[key] = LongTensor(example[key])

        return example

    def __len__(self):
        return len(self.data)

def train():
    train_dataset = load_dataset('json', data_files='data/train.json', split='train')
    val_dataset = load_dataset('json', data_files='data/val.json', split='train')
    test_dataset = load_dataset('json', data_files='data/test.json', split='train')
    print(train_dataset)
    print(test_dataset)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    train_inputs = tokenizer(train_dataset['utterance'], padding='longest', truncation=True, return_tensors="pt")
    train_outputs = tokenizer(train_dataset['act_str'], padding='longest', truncation=True, return_tensors="pt")

    val_inputs = tokenizer(val_dataset['utterance'], padding='longest', truncation=True, return_tensors="pt")
    val_outputs = tokenizer(val_dataset['act_str'], padding='longest', truncation=True, return_tensors="pt")

    # Define a PyTorch Dataset
    # class MyDataset(Dataset):
    #     def __init__(self, inputs, outputs):
    #         self.inputs = inputs
    #         self.outputs = outputs
    #     def __getitem__(self, idx):
    #         return {key: tensor[idx] for key, tensor in self.inputs.items()}, self.outputs['input_ids'][idx]
    #     def __len__(self):
    #         return self.inputs['input_ids'].shape[0]
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels['input_ids'][idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = MyDataset(train_inputs, train_outputs)
    val_dataset = MyDataset(val_inputs, val_outputs)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'utterance', 'act_str'])

    batch_size = 128
    # Define the training arguments and trainer
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,              
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=batch_size,   
        warmup_steps=40,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        learning_rate=2e-5,
        evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    eval_result = trainer.evaluate()
    print(eval_result)

    # Save the model
    trainer.save_model('./results')

    test_encodings = tokenizer([instance['utterance'] for instance in test_dataset], padding='longest', truncation=True, return_tensors="pt")
    test_labels = tokenizer([instance['act_str'] for instance in test_dataset], padding='longest', truncation=True, return_tensors="pt")
    test_dataset = MyDataset(test_encodings, test_labels)
    print(test_dataset)
    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions

    # Convert to a tensor if it's a numpy array
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Move to CPU if it's on a GPU
    if logits.is_cuda:
        logits = logits.cpu()
    # Apply softmax to the logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Convert probabilities to token ids (argmax)
    predicted_ids = torch.argmax(probs, dim=-1)
    # Decode the token ids into text
    predicted_sentences = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    for sentence in predicted_sentences:
        print(sentence)
    
    # outputs = model.generate(**predicted, max_new_tokens=100)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
