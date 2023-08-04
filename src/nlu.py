import argparse
import numpy as np
import torch
import warnings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from utils.constant import ID_TO_SLOT, ID_TO_ACT


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('utterance', type=str)
    args = parser.parse_args()
    text = args.utterance

    model_act = AutoModelForSequenceClassification.from_pretrained("models/model_act_100")
    model_slot = AutoModelForSequenceClassification.from_pretrained("models/model_slot_100")
    model_value = AutoModelForSeq2SeqLM.from_pretrained("models/model_value_100")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_value = AutoTokenizer.from_pretrained("google/flan-t5-base")

    preds = []

    # Act-Predictor
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

    for act_label in predicted_act_labels:
        # Slot-Predictor
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

        for slot_label in predicted_slot_labels:
            # Value-Predictor
            text_slot = f"{text} [SEP] {act_label} [SEP] {slot_label}"
            tokenized_text = tokenizer_value(text_slot, padding='longest', truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model_value.generate(**tokenized_text)
            logits = outputs
            pred_tokens = [tokenizer_value.decode(ids, skip_special_tokens=True) for ids in logits]
            pred_tokens_str = ' '.join(pred_tokens)

            preds.append((act_label, slot_label, pred_tokens_str))

    preds_text = []
    for pred in preds:
        preds_text.append(f"(act={pred[0]}, slot={pred[1]}, value={pred[2]})")
    
    print("".join(preds_text))

