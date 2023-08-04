
from utils.constant import MAX_TOKENS, DATA_PATH, SLOT_TO_ID, ID_TO_SLOT

import json
import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch import LongTensor
from sklearn.metrics import f1_score, accuracy_score


model = AutoModelForSequenceClassification.from_pretrained("model_slot")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with open('data/test_slot.json', 'r') as f:
    data = json.load(f)

for test in data:
    text = test['utterance']
    tokenized_text = tokenizer(text, padding='longest', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_text)

    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [ID_TO_SLOT[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(f"{text}: predicted:{predicted_labels}, actual:{test['slot_labels']}")