import argparse
import json
import os

import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

import utils.prompt as prompt

parser = argparse.ArgumentParser(description='T5')
parser.add_argument('-c', '--cot', action='store_true')
parser.add_argument('-e', '--exp', action='store_true')
parser.add_argument('-p', '--posthoc', action='store_true')
parser.add_argument('-g', '--graph', default='')
parser.add_argument('-m', '--model', default='t5-3b')
parser.add_argument('-E', '--epoch', default=3, type=int)
args = parser.parse_args()

# Initialize the T5 tokenizer
cot = args.cot
exp = args.exp
posthoc = args.posthoc
graph = args.graph
model_name = args.model
tokenizer = T5Tokenizer.from_pretrained(model_name, truncation=True, model_max_length=1024)

file_suffix = '_' + model_name
if cot:
    file_suffix += '_cot'
if exp:
    file_suffix += '_exp'
if posthoc:
    file_suffix += '_posthoc'
if '' != graph:
    file_suffix += '_' + graph
print(file_suffix)

# convert json to dataset format
def convert(data):
    message = prompt.compose_prompt(data, tokenizer, cot=cot, exp=exp, posthoc=posthoc, graph=graph)
    if [] == data['answer']:
        answer = ''
    else:
        answer = prompt.compose_answer(data, cot, exp, posthoc)
    return {"input": message, "target": answer}

def convert_file(split):
    if os.path.exists(f'../data/meqa/converted_{split}{file_suffix}.json'):
        return
    with open(f'../data/meqa/collected_{split}.json', 'r') as f:
        data = json.load(f)
    converted_data = [convert(item) for item in data if '' != item['question']]
    with open(f'../data/meqa/converted_{split}{file_suffix}.json', 'w') as f:
        json.dump(converted_data, f, indent=2)
    print(f'{split} done')

convert_file('train')
convert_file('dev')
convert_file('test')

# Load data from JSON files
train_dataset = load_dataset('json', data_files=f'../data/meqa/converted_train{file_suffix}.json')['train']
val_dataset = load_dataset('json', data_files=f'../data/meqa/converted_dev{file_suffix}.json')['train']
test_dataset = load_dataset('json', data_files=f'../data/meqa/converted_test{file_suffix}.json')['train']

print(train_dataset[0])

def tokenize_function(batch):
    # Tokenize input and target in batches
    inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=1024, return_tensors="pt")
    targets = tokenizer(batch['target'], padding='max_length', truncation=True, max_length=1024, return_tensors="pt")

    # Shift the target sequences to the right
    decoder_input_ids = targets["input_ids"].clone()
    decoder_input_ids = torch.cat([torch.full_like(decoder_input_ids[:, :1], tokenizer.pad_token_id), decoder_input_ids[:, :-1]], dim=-1)

    # Add decoder_input_ids and labels to the returned dictionary
    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["labels"] = targets["input_ids"]
    
    return inputs

# Tokenize datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Initialize T5 model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Training arguments
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=1,
    num_train_epochs=args.epoch,
    logging_dir='../logs/',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    output_dir='../results/',
    push_to_hub=False,
    fp16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained(f"../results/t5_finetuned_model{file_suffix}")
tokenizer.save_pretrained(f"../results/t5_finetuned_model{file_suffix}")

# # Evaluate on the test set
# test_results = trainer.predict(test_tokenized).predictions
# test_preds_ids = test_results.argmax(dim=-1)
# test_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in test_preds_ids]
# test_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in test_tokenized["labels"]]

# # If the `labels` are returned as PyTorch tensors, convert them to a list first
# if hasattr(test_tokenized["labels"], "tolist"):
#     test_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in test_tokenized["labels"].tolist()]

# precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='micro')

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
