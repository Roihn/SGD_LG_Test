import json

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
            acts_str = []
            acts = []
            slots = []
            values = []
            for frame in turn['frames']:
                for action in frame['actions']:
                    # Extract the utterance and act
                    act = action['act']
                    slot = action['slot']
                    value = action['values']
                    act_str = f"(act={act}, slot={slot or 'None'}, value={', '.join(value) or 'None'})"
                    acts_str.append(act_str)
                    acts.append(act)
                    slots.append(slot or 'None')
                    values.append(', '.join(value) or 'None')
            
            # if len(acts) > 1:
            #     print(''.join(acts_str))
            flattened_data['utterance'].append(utterance)
            flattened_data['act_str'].append(''.join(acts_str))
            flattened_data['act_labels'].append(acts)
            flattened_data['slot_labels'].append(slots)
            flattened_data['value_labels'].append(values)

    return flattened_data


if __name__ == "__main__":
    with open('dialogues_001.json') as f:
        data = json.load(f)
    
    flattened_data = flatten_data(data)
    # store the flattened data
    with open('processed_dialogues_001.json', 'w') as f:
        json.dump(flattened_data, f, indent=4)
