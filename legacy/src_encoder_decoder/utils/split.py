import json
from sklearn.model_selection import train_test_split

# Load your JSON data
with open('processed_dialogues_001.json', 'r') as f:
    data = json.load(f)

# Assuming your data is a list of objects, you could convert it to a pandas DataFrame for easier manipulation
import pandas as pd
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42)

# Split the data into training, validation, and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)  # 80% training, 20% test
train, val = train_test_split(train, test_size=0.25, random_state=42)  # Then split training into 75% training, 25% validation, for a 60-20-20 split overall

# You can then save these DataFrames to JSON files if needed:
train.to_json('data/train.json', orient='records')
val.to_json('data/val.json', orient='records')
test.to_json('data/test.json', orient='records')