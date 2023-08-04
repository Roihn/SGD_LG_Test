import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

model = AutoModelForSeq2SeqLM.from_pretrained("models/model_value_100")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

with open('data/test_value.json', 'r') as f:
    data = json.load(f)

TP = 0
FP = 0
FN = 0

for test in tqdm(data):
    text = test['utterance']
    tokenized_text = tokenizer(text, padding='longest', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**tokenized_text)
    logits = outputs
    pred_tokens = [tokenizer.decode(ids, skip_special_tokens=True) for ids in logits]
    pred_tokens = ' '.join(pred_tokens)
    if pred_tokens is None:
        print("warning", text)
    true_tokens = test['value_labels']
    if true_tokens == pred_tokens:
        TP += 1
    else:
        if pred_tokens is None:
            FN += 1
        FP += 1

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"recall: {recall}, precision: {precision}, f1: {f1}")
