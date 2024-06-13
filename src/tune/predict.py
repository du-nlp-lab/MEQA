import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, EvalPrediction
from typing import Dict

import evaluator.hotpot_evaluate as hotpot_evaluate
import utils.prompt as prompt

parser = argparse.ArgumentParser(description='T5')
parser.add_argument('-c', '--cot', action='store_true')
parser.add_argument('-e', '--exp', action='store_true')
parser.add_argument('-p', '--posthoc', action='store_true')
parser.add_argument('-g', '--graph', action='store_true')
parser.add_argument('-m', '--model', default='t5-3b')
args = parser.parse_args()

cot = args.cot
exp = args.exp
posthoc = args.posthoc
graph = args.graph
model_name = args.model

file_suffix = '_' + model_name
if cot:
    file_suffix += '_cot'
if exp:
    file_suffix += '_exp'
if posthoc:
    file_suffix += '_posthoc'
if graph:
    file_suffix += '_graph'
print(file_suffix)

#with open('../../complex-event-qa/data/example.json', 'r') as f:
#    example = json.load(f)
#
#def convert_to_t5_format(data):
#    example_prompt = prompt.compose_example(data)
#    message = prompt.compose_prompt(data, example_prompt)
#    return {"input": t5_input, "target": data["answer"]}
#
#if not os.path.exists('collected_test_data.json'):
#    # Load the data
#    with open("../../complex-event-qa/data/collected_test.json", "r") as file:
#        dataset = json.load(file)
#    
#    # Convert the dataset to T5 format and filter out items with empty questions
#    converted_dataset = [convert_to_t5_format(item) for item in dataset if item["question"] != ""]
#    
#    # Save the converted and split data back to new JSON files
#    
#    with open("../../complex-event-qa/data/collected_test_data.json", "w") as file:
#        json.dump(converted_dataset, file, indent=4)
#
#    print("Data converted and saved.")
#else:
#    with open("collected_test_data.json", "r") as file:
#        json.load(file)
test_dataset = load_dataset('json', data_files=f'../data/meqa/converted_test{file_suffix}.json')['train']

# Load the saved model and tokenizer
saved_model_path = f"../results/t5_finetuned_model{file_suffix}"
loaded_model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
#loaded_tokenizer = T5Tokenizer.from_pretrained(saved_model_path, truncation=True, model_max_length=1024)
loaded_tokenizer = T5Tokenizer.from_pretrained(model_name, truncation=True, model_max_length=1024)

# Training arguments
training_args = TrainingArguments(
    do_predict=True,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_dir='../logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    output_dir='../results',
    push_to_hub=False,
    fp16=True,
)

#tokenizer = T5Tokenizer.from_pretrained(model_name, truncation=True, model_max_length=1024)
tokenizer = loaded_tokenizer
def tokenize_function(batch):
    inputs = tokenizer(batch['input'], padding='max_length', truncation=True, return_tensors="pt", max_length=1024)
    targets = tokenizer(batch['target'], padding='max_length', truncation=True, return_tensors="pt", max_length=1024)

    decoder_input_ids = targets["input_ids"].clone()
    decoder_input_ids = torch.cat([torch.full_like(decoder_input_ids[:, :1], tokenizer.pad_token_id), decoder_input_ids[:, :-1]], dim=-1)
    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["labels"] = targets["input_ids"]
    return inputs

test_tokenized = test_dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model=loaded_model,
    args=training_args,
    tokenizer=loaded_tokenizer
)

prediction = trainer.predict(test_tokenized).predictions[0]
predicted_token_ids = np.argmax(prediction, axis=-1)  # Shape should now be (83, 512)
# np.save('predictions.npy', predicted_token_ids)

# Decode token IDs to text
decoded_answers = [loaded_tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in predicted_token_ids]

# Combine _id with the decoded answers
# combined_data = [{"_id": entry["_id"], "answer": answer} for entry, answer in zip(test_dataset, decoded_answers)]
combined_data = [{"answer": answer} for answer in decoded_answers]

# Save to a JSON file
with open(f"../results/combined_results{file_suffix}.json", "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=4)
        
total_f1, total_prec, total_recall = [], [], []
for pred_ans, gold_ans in zip(decoded_answers, test_dataset):
    pred_ans = prompt.extract_answer(pred_ans, cot, exp, posthoc)
    if str != type(pred_ans):
        pred_ans, exp = pred_ans
    gold_ans = gold_ans['target']
    f1, prec, recall = hotpot_evaluate.f1_score(pred_ans.lower(), gold_ans.lower())
    total_f1.append(f1)
    total_prec.append(prec)
    total_recall.append(recall)

total_f1 = np.sum(total_f1) / len(total_f1)
total_prec = np.sum(total_prec) / len(total_prec)
total_recall = np.sum(total_recall) / len(total_recall)
print(total_f1)
print(total_prec)
print(total_recall)

#true_labels_tokenized = tokenizer(test_dataset['target'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
#true_labels_ids = true_labels_tokenized['input_ids'].numpy()
#
## Flatten the arrays
#true_labels_flat = true_labels_ids.reshape(-1)
#predicted_flat = predicted_token_ids.reshape(-1)
#
## Mask out padding tokens from true labels and apply the same mask to predicted labels
#mask = true_labels_flat != tokenizer.pad_token_id
#
#true_labels_masked = true_labels_flat[mask]
#predicted_masked = predicted_flat[mask]
#
#from sklearn.metrics import accuracy_score
#
#accuracy = accuracy_score(true_labels_masked, predicted_masked)
#precision, recall, f1, _ = precision_recall_fscore_support(true_labels_masked, predicted_masked, average='weighted')
#
#print(f"Accuracy: {accuracy}")
#print(f"Precision: {precision}")
#print(f"Recall: {recall}")
#print(f"F1 Score: {f1}")

