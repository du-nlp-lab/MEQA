import argparse
import json
import re
import time

import numpy as np
import tiktoken
from tqdm import tqdm

import evaluator.hotpot_evaluate as hotpot_evaluate
import evaluator.explanation_evaluate as explanation_evaluate
import utils.prompt as prompt
import utils.openai_api as openai_api

parser = argparse.ArgumentParser(description='GPT')
parser.add_argument('-c', '--cot', action='store_true')
parser.add_argument('-e', '--exp', action='store_true')
parser.add_argument('-p', '--posthoc', action='store_true')
parser.add_argument('-g', '--graph', default='')
args = parser.parse_args()

with open('../data/meqa/collected_test.json', 'r') as f:
    data = json.load(f)
    print(len(data))

with open('../data/meqa/example.json', 'r') as f:
    example = json.load(f)

cot = args.cot
exp = args.exp
posthoc = args.posthoc
graph = args.graph
model_name = 'gpt-3.5-turbo-1106'
file_suffix = ''
if cot:
    file_suffix += '_cot'
if exp:
    file_suffix += '_exp'
if posthoc:
    file_suffix += '_posthoc'
if '' != graph:
    file_suffix += '_' + graph

example = prompt.compose_example(example, cot=cot, exp=exp, posthoc=posthoc, graph=graph)

tokenizer = tiktoken.encoding_for_model(model_name)
total_f1 = []
total_prec = []
total_recall = []
exp_f1 = []
exp_prec = []
exp_recall = []
for d in tqdm(data):
    message = prompt.compose_prompt(d, tokenizer, example, cot=cot, exp=exp, posthoc=posthoc, graph=graph)
    if type(d['answer']) is str:
        golden_answers = [ans.split('@')[0].strip() for ans in d['answer'].split(',')]
    elif type(d['answer']) is list:
        golden_answers = [ans.split('@')[0].strip() for ans in d['answer']]
#    print(message)
#    input()
    pred = openai_api.call(message)
#    print(pred)
#    input()
    d['prediction_ori'] = pred
    pred = prompt.extract_answer(pred, cot=cot, exp=exp, posthoc=posthoc)
    explanation = None
    if str != type(pred):
        pred, explanation = pred
    if cot:
        pred = '. '.join(pred.split('. ')[-2:])
    d['predicted_answer'] = pred
    if explanation is not None:
        d['predicted_explanation'] = explanation
#    print('===================')
#    print(pred, '-', golden_answers)
#    if explanation is not None:
#        print(explanation, '\n', d['explanation'])
#    input()

    for answer in golden_answers:
        f1, prec, recall = hotpot_evaluate.f1_score(pred.lower(), answer.lower())
        total_f1.append(f1)
        total_prec.append(prec)
        total_recall.append(recall)

#    if explanation is not None:
#        f1, prec, recall = explanation_evaluate.f1_score(explanation, d['explanation'])
#        exp_f1.append(f1)
#        exp_prec.append(prec)
#        exp_recall.append(recall)

with open(f'../results/predictions{file_suffix}.json', 'w') as f:
    json.dump(data, f, indent=2)

total_f1 = np.sum(total_f1) / len(total_f1)
total_prec = np.sum(total_prec) / len(total_prec)
total_recall = np.sum(total_recall) / len(total_recall)
print(total_f1)
print(total_prec)
print(total_recall)

#if 0 < len(exp_f1):
#    exp_f1 = np.sum(exp_f1) / len(exp_f1)
#    exp_prec = np.sum(exp_prec) / len(exp_prec)
#    exp_recall = np.sum(exp_recall) / len(exp_recall)
#    print(exp_f1)
#    print(exp_prec)
#    print(exp_recall)

