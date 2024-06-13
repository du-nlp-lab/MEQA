import json

import numpy as np
import spacy
from tqdm import tqdm

import evaluator.hotpot_evaluate as hotpot_evaluate

with open('../results/predictions_cot_golden-event.json', 'r') as f:
    data = json.load(f)

nlp = spacy.load("en_core_web_lg")

total_f1 = []
total_prec = []
total_recall = []
for d in tqdm(data):
    pred_answer = d['predicted_answer']
#    sentences = pred_answer.split('. ')[-2:]
#    pred_answer = '. '.join(sentences)
    sentences = [str(sent).strip() for sent in nlp(pred_answer).sents][-2:]
    pred_answer = ' '.join(sentences)

    if type(d['answer']) is str:
        golden_answers = [ans.split('@')[0].strip() for ans in d['answer'].split(',')]
    elif type(d['answer']) is list:
        golden_answers = [ans.split('@')[0].strip() for ans in d['answer']]

    for answer in golden_answers:
        f1, prec, recall = hotpot_evaluate.f1_score(pred_answer.lower(), answer.lower())
        total_f1.append(f1)
        total_prec.append(prec)
        total_recall.append(recall)

total_f1 = np.sum(total_f1) / len(total_f1)
total_prec = np.sum(total_prec) / len(total_prec)
total_recall = np.sum(total_recall) / len(total_recall)
print(total_f1)
print(total_prec)
print(total_recall)

