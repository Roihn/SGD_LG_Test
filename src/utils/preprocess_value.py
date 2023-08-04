import json


def flatten_data(data):
    flattened_data = {
        'utterance': [],
        'value_labels': [],
    }
    for dialogue in data:
        for turn in dialogue['turns']:
            utterance = turn['utterance']
            for frame in turn['frames']:
                for action in frame['actions']:
                    utterance_act = f"{utterance} [SEP] {action['act']} [SEP] {action['slot'] or 'None'}"
                    flattened_data['utterance'].append(utterance_act)
                    flattened_data['value_labels'].append(', '.join(action['values']) or 'None')

    return flattened_data


if __name__ == "__main__":
    with open('dialogues_001.json') as f:
        data = json.load(f)
    flattened_data = flatten_data(data)
    # store the flattened data
    with open('processed_dialogues_001_value.json', 'w') as f:
        json.dump(flattened_data, f, indent=4)
