
from utils.constant import MAX_TOKENS, DATA_PATH, SLOT_TO_ID, ID_TO_SLOT, ID_TO_ACT, ACT_TO_ID

import json
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from tqdm import tqdm
from copy import deepcopy as copy

model_act = AutoModelForSequenceClassification.from_pretrained("model_act")
model_slot = AutoModelForSequenceClassification.from_pretrained("model_slot")
model_value = AutoModelForSeq2SeqLM.from_pretrained("model_value")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_value = AutoTokenizer.from_pretrained("google/flan-t5-base")

with open('data/test.json', 'r') as f:
    data = json.load(f)

TP = 0
FP = 0
FN = 0
for test in data:
    flag = True
    preds = []
    text = test['utterance']
    print(text)
    tokenized_text = tokenizer(text, padding='longest', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_act(**tokenized_text)
        
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_act_labels = [ID_TO_ACT[idx] for idx, label in enumerate(predictions) if label == 1.0]
    if len(predicted_act_labels) == 0: 
        print(f"FN: {test['utterance']}: {predicted_act_labels}, Not able to predict act")
        FN += 1
        continue
    for act_label in predicted_act_labels:
        text_act = text + " " + act_label
        tokenized_text = tokenizer(text_act, padding='longest', truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model_slot(**tokenized_text)

        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_slot_labels = [ID_TO_SLOT[idx] for idx, label in enumerate(predictions) if label == 1.0]
        if len(predicted_slot_labels) == 0:
            print(f"FN: {test['utterance']}: {predicted_slot_labels}, cur_act: {act_label}, Not able to predict slot. actual:{test['act_str']}")
            FN += 1
            flag = False
            break
        
        for slot_label in predicted_slot_labels:
            text = f"{test['utterance']} [SEP] {act_label} [SEP] {slot_label}"
            tokenized_text = tokenizer_value(text, padding='longest', truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model_value.generate(**tokenized_text)
            logits = outputs
            pred_tokens = [tokenizer_value.decode(ids, skip_special_tokens=True) for ids in logits]
            if len(pred_tokens) == 0:
                FN += 1
                flag = False
                break
            pred_tokens_str = ' '.join(pred_tokens)
            if slot_label == 'cuisine':
                print(pred_tokens, pred_tokens_str)

            preds.append((act_label, slot_label, pred_tokens_str))

        if not flag:
            break
    
    if not flag:
        continue
    # Evaluate the predictions
    tmp_test = copy(test)
    for pred in preds:
        pred_matched = False
        for i in range(len(tmp_test['act_labels'])):
            if pred[0] == tmp_test['act_labels'][i] and pred[1] == tmp_test['slot_labels'][i] and pred[2] == tmp_test['value_labels'][i]:
                # remove the matched act, slot, value
                tmp_test['act_labels'].pop(i)
                tmp_test['slot_labels'].pop(i)
                tmp_test['value_labels'].pop(i)
                pred_matched = True
                break
        if not pred_matched:
            FP += 1
            flag = False
            print(f"FP in the loop: {test['utterance']}: {preds}, actual:{test}")
            break
    
    if not flag:
        continue

    if len(tmp_test['act_labels']) > 0:
        FP += 1
        print(f"FP: {test['utterance']}: {preds}, actual:{test['act_str']}")
    else:
        TP += 1
    

    # print(f"{test['utterance']}: {preds}, actual:{test['act_str']}")

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"recall: {recall}, precision: {precision}, f1: {f1}")
    