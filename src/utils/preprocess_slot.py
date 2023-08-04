import json
from collections import defaultdict

def flatten_data(data):
    flattened_data = {
        'utterance': [],
        'slot_labels': [],
    }
    for dialogue in data:
        for turn in dialogue['turns']:
            utterance = turn['utterance']
            for frame in turn['frames']:
                actions = list(frame['actions'])
                action_slot_dict = defaultdict(list)
                for action in actions:
                    action_slot_dict[action['act'] or "None"].append(action['slot'] or "None")
                for act in action_slot_dict.keys():
                    flattened_data['utterance'].append(f"{utterance} [SEP] {act}")
                    flattened_data['slot_labels'].append(action_slot_dict[act])
                    print(f"{utterance} {act} {action_slot_dict[act]}")

    return flattened_data


if __name__ == "__main__":
    with open('dialogues_001.json') as f:
        data = json.load(f)
    
    flattened_data = flatten_data(data)
    # store the flattened data
    with open('processed_dialogues_001_slot.json', 'w') as f:
        json.dump(flattened_data, f, indent=4)
