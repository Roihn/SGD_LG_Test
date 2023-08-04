
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, TrainingArguments, Trainer

def train(proportion=100):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    train_dataset = load_dataset('json', data_files=f'data/{proportion}/train_value.json', split='train')
    val_dataset = load_dataset('json', data_files=f'data/{proportion}/test_value.json', split='train')
    print(train_dataset)
    def preprocess_data(examples):
        inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        targets = tokenizer(examples['value_labels'], padding='max_length', truncation=True, max_length=32, return_tensors='pt').input_ids
        inputs['labels'] = targets
        return inputs
    
    train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    # train_dataset.set_format('torch')
    val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names)
    # val_dataset.set_format('torch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_size = 4
    metrics = "f1"
    training_args = TrainingArguments(
        output_dir='./output',
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=40, 
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=2e-5,
        metric_for_best_model=metrics,
        evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    
    trainer.save_model(f"models/model_value_{proportion}")