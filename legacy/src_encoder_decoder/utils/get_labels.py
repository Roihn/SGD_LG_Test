import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset

def flatten_data(data):
    flattened_data = {
        'utterance': [],
        'act': []
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
            flattened_data['act'].append(''.join(acts))
    return flattened_data


if __name__ == "__main__":
    with open('dialogues_001.json') as f:
        data = json.load(f)

    # flatten the data
    acts = set()
    slots = set()
    values = set()
    utterances = []

    for dialogue in data:
        for turn in dialogue['turns']:
            utterances.append(turn['utterance'])
            for frame in turn['frames']:
                # print(frame)
                for action in frame['actions']:
                    # print(action)
                    acts.add(action['act'])
                    slots.add(action['slot'])
                    if action['values'] is not None:
                        if action['slot'] == 'cuisine':
                            print("####", turn['utterance'], action['slot'], action['values'])
                        if len(action['values']) > 1:
                            assert action['slot'] == 'cuisine'
                            print(turn['utterance'], action['slot'], action['values'])
                        for value in action['values']:
                            values.add(value)
    
    print(acts)
    print(slots)
    print(values)


    flattened_data = flatten_data(data)
    dataset = Dataset.from_dict(flattened_data)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    print(dataset)
    tokenized_utterances = tokenizer(utterances, truncation=False, padding=False)

    lengths = [len(utterance) for utterance in tokenized_utterances['input_ids']]

    # Find the maximum length
    max_tokens = max(lengths)

    print(max_tokens)

    act_label_map = {label: idx for idx, label in enumerate(acts)}
    slot_label_map = {label: idx for idx, label in enumerate(slots)}


    print(act_label_map)
    print(slot_label_map)


    act_label_map_inv = {v: k for k, v in act_label_map.items()}
    slot_label_map_inv = {v: k for k, v in slot_label_map.items()}

    print(act_label_map_inv)
    print(slot_label_map_inv)