from src.model import DialogueActSlotValueModel
from src.utils.constant import MAX_TOKENS, DATA_PATH, SLOT_TO_ID, ID_TO_SLOT

import json
import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch import LongTensor
from sklearn.metrics import f1_score, accuracy_score

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    for i in range(len(predictions)):
        predicted_labels = [ID_TO_SLOT[idx] for idx, label in enumerate(predictions[i]) if label == 1.0]
        print([ID_TO_SLOT[j] for j, val in enumerate(labels[i]) if val == 1], predicted_labels)

    
    f1_micro_averge = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        'f1': f1_micro_averge,
        'accuracy': accuracy
    }
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(preds, p.label_ids)
    return result

def train():
    train_dataset = load_dataset('json', data_files='data/train_slot.json', split='train')
    val_dataset = load_dataset('json', data_files='data/test_slot.json', split='train')
    print(train_dataset)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                               problem_type="multi_label_classification", 
                                                               num_labels=len(SLOT_TO_ID), 
                                                               id2label=ID_TO_SLOT,
                                                               label2id=SLOT_TO_ID)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    

    def preprocess_data(examples):
        text = examples['utterance']
        encoding = tokenizer(text, padding='longest', truncation=True, return_tensors="pt")

        labels_matrix = np.zeros((len(text), len(SLOT_TO_ID)))
        for i, example_labels in enumerate(examples['slot_labels']):
            print(example_labels, examples['utterance'][i])
            for label in example_labels:
                if label in SLOT_TO_ID:
                    labels_matrix[i, SLOT_TO_ID[label]] = 1

        encoding['labels'] = labels_matrix.tolist()
        return encoding


    train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'utterance', 'act_str'])

    batch_size = 4
    metric_name = "f1"
    # Define the training arguments and trainer
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=10,              
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=batch_size,   
        warmup_steps=40,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        learning_rate=2e-5,
        metric_for_best_model=metric_name,
        evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    eval_result = trainer.evaluate()
    print(eval_result)
    trainer.save_model("model_slot")

    texts = ["I am looking to eat somewhere [SEP] INFORM_INTENT", "I have your appointment set up. They do serve alcohol. [SEP] INFORM", "I've found 1 restaurant that could work. It's Lovina in Calistoga. [SEP] INFORM_COUNT", "What's their phone number? [SEP] REQUEST"]

    for text in texts:

        encoding = tokenizer(text, padding='longest', truncation=True, return_tensors="pt")
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

        outputs = trainer.model(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [ID_TO_SLOT[idx] for idx, label in enumerate(predictions) if label == 1.0]
        print(f"{text}: {predicted_labels}")