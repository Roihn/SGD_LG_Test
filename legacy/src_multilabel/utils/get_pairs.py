import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset

if __name__ == "__main__":
    with open('dialogues_001.json') as f:
        data = json.load(f)

    # flatten the data
    act_slot_pairs = set()

    for dialogue in data:
        for turn in dialogue['turns']:
            for frame in turn['frames']:
                # print(frame)
                for action in frame['actions']:
                    # print(action)
                    act_slot_pairs.add((action['act'], action['slot']))
    
    print(act_slot_pairs)
    print(len(act_slot_pairs))

    
    label_map = {label: idx for idx, label in enumerate(act_slot_pairs)}

    
    label_map_inv = {v: k for k, v in label_map.items()}

    print(label_map)
    print(label_map_inv)