from src.model import DialogueActSlotValueModel
from src.utils.constant import MAX_TOKENS, DATA_PATH, ACT_TO_ID, SLOT_TO_ID, ID_TO_ACT, ID_TO_SLOT

import json
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, BertTokenizer
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


def flatten_data(data):
    flattened_data = {
        'utterance': [],
        'act_str': [],
        'act_labels': [],
        'slot_labels': [],
        'value_labels': [],
    }
    for dialogue in data:
        for turn in dialogue['turns']:
            utterance = turn['utterance']
            acts = []
            for frame in turn['frames']:
                for action in frame['actions']:
                    # Extract the utterance and act
                    act = action['act']
                    slot = action['slot']
                    value = action['values']
                    act_str = f"(act={act}, slot={slot}, value={value})"
                    acts.append(act_str)
                    # Add them to the flattened data
            # flattened_data.append({'utterance': utterance, 'act': ''.join(acts)})
            flattened_data['utterance'].append(utterance)
            # flattened_data['act'].append(''.join(acts))
            flattened_data['act_str'].append(acts[0])
            flattened_data['act_labels'].append(ACT_TO_ID[act])
            flattened_data['slot_labels'].append(SLOT_TO_ID[slot])
            flattened_data['value_labels'].append(0)
    return flattened_data


def train():
    with open(DATA_PATH) as f:
        data = json.load(f)

    train_dataset = load_dataset('json', data_files='data/train.json', field='data')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(train_dataset)

    def encode_utterance(examples):
        return tokenizer(examples['utterance'], truncation=True, padding='max_length', max_length=MAX_TOKENS)

    def encode_act(examples):
        return tokenizer(examples['act'], truncation=True, padding='max_length', max_length=MAX_TOKENS)
    
    def encode(examples):
        encoded_input = tokenizer(examples['utterance'], truncation=True, padding='max_length', max_length=MAX_TOKENS)
        examples['act_labels'] = [int(label) for label in examples['act_labels']]
        examples['slot_labels'] = [int(label) for label in examples['slot_labels']]
        examples['value_labels'] = [int(label) for label in examples['value_labels']]

        return {**encoded_input, **examples}

    train_dataset = train_dataset.map(encode, batched=True)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'utterance', 'act_labels', 'slot_labels', 'value_labels'])

    num_acts = len(ACT_TO_ID)
    num_slots = len(SLOT_TO_ID)
    num_values = MAX_TOKENS

    model = DialogueActSlotValueModel(num_acts, num_slots, num_values)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Loss function
    loss_fn = CrossEntropyLoss()

    num_epochs = 10
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            # Each batch contains input_ids, attention_mask, and labels
            # Move batch to the same device as the model
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            act_labels = batch['act_labels'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            value_labels = batch['value_labels'].to(device)

            # Forward pass
            act_logits, slot_logits, value_logits = model(input_ids, attention_mask)

            # Compute loss
            act_loss = loss_fn(act_logits, act_labels)
            slot_loss = loss_fn(slot_logits, slot_labels)
            value_loss = loss_fn(value_logits, value_labels)
            loss = act_loss + slot_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}")
    
    # Save the model
    torch.save(model.state_dict(), 'model.pt')
