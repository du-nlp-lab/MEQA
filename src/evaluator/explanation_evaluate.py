import json
import re

import numpy as np
import spacy
from tqdm import tqdm

import utils.openai_api as openai_api

nlp = spacy.load("en_core_web_lg")

def exp_exact_match_prompt(exp, golden_exp):
    message = ('Please determine whether the following question-answer'
               'pairs have the same semantic meanings. '
               'Please only output "yes" or "no":')
    message = '\n'.join([message, exp, golden_exp])
    return message

def f1_score(exps, golden_exps):
    while len(exps) < len(golden_exps):
        exps.append('')
    while len(exps) > len(golden_exps):
        golden_exps.append('')

    same = 0
    for exp, golden_exp in zip(exps, golden_exps):
        if '' == exp or '' == golden_exp:
            continue
        message = exp_exact_match_prompt(exp, golden_exp)
#        print(message)
        pred = openai_api.call(message)
#        print(pred)
#        input()
        if 'yes' == pred:
            same += 1

    if 0 == same:
        return 0, 0, 0

    precision = same / len(exps)
    recall = same / len(golden_exps)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall

def exp_consistency_prompt(pair, first=False):
    if first:
        message = (pair[0],
                   '\n\nPlease determine whether following question',
                   'is a sub-question of the previous question?\n',
                   pair[1].split('?')[0] + '?',
                   '\n\nPlease only output "yes" or "no": '
                   )
        message = ''.join(message)
    else:
        message = (pair[0],
                   '\n\nPlease determine whether the following question ',
                   'has logical contradiction to the previous content?\n',
                   pair[1].split('?')[0] + '?',
                   '\nPlease only output "yes" or "no": '
                   )
        message = ''.join(message)
    return message

def cot_consistency_prompt(pair, first=False):
    if first:
        message = (pair[0],
                   '\n\nPlease determine whether the following sentence ',
                   'answers or matches part of the previous question?\n',
                   pair[1],
                   '\n\nPlease output "yes" or "no" first. If "yes", please output the answer or match: '
                   )
        message = ''.join(message)
    else:
        message = (
            "You are an expert on recognizing logical contradictions. ",
            "A logical contradiction is the conjunction of a statement S and its denial, not-S.\n\n",
            "Please determine whether the following sentences contain a logical contradiction:\n",
            f"1. {pair[0]}\n",
            f"2. {pair[1]}\n",
            "For instance, the example, 'the document doesn't provide xxx, where xxx may come from a question', ",
            "represents a logical contradiction.\n",
            "Please output 'yes' or 'no' first. If 'yes', explain where the contradiction lies: "
        )
        message = ''.join(message)

    return message

def reasoning_steps_consistency(question, exps, flag):
    consistency = []
    question = 'Question: ' + question
    for i in range(0, len(exps)):
        if '' == exps[i]:
            continue
        history = '\n'.join([question, *exps[:i]])
        pair = [history, exps[i]]
        if flag:
            message = exp_consistency_prompt(pair, 0==i)
        else:
            message = cot_consistency_prompt(pair, 0==i)
#        print(message)
        pred = openai_api.call(message)
#        print(pred)
#        input()
        if 'yes' == pred[:3].lower():
            consistency.append(1 if 0 == i else 0)
        else:
            consistency.append(0 if 0 == i else 1)

    return consistency

def cal_consistency(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    total_consistency = []
    avg_consistency = []
    for d in tqdm(data):
        if 'predicted_explanation' in d:
            pred_exp = d['predicted_explanation']
        else:
#            pred_exp = d['prediction_ori'].split('. ')[-2:]
            pred_exp = [str(sent).strip() for sent in nlp(d['prediction_ori']).sents][-2:]
        consistency = reasoning_steps_consistency(d['question'], pred_exp, 'predicted_explanation' in d)
        total_consistency.append(consistency)
        if 0 == len(consistency):
            avg_consistency.append(0)
        else:
            avg_consistency.append(np.sum(consistency) / len(consistency))
        d['pred_consistency'] = consistency
        d['pred_consistency_score'] = avg_consistency[-1]
    print(np.sum(avg_consistency) / len(avg_consistency))

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)

def incompleteness(d):
    if 'predicted_explanation' in d:
        if 0 == len(d['predicted_explanation']):
            return 0, 0, 0
    else:
        if 0 == len(d['prediction_ori']):
            return 0, 0, 0

#    print('golden_explanation: ', d['explanation'])

    # extract golden full graph
    events = d['events']
    triples = []
    # connect all arguments and events
    for event in events:
        trigger = event['trigger']['text']
        for argument in event['arguments']:
            triple = [trigger, argument['role'], argument['text']]
            triples.append(triple)
    # for strategy 1, connect events by event relations
    if 'relation' in d:
        trigger1 = events[0]['trigger']['text']
        trigger2 = events[1]['trigger']['text']
        triple = [trigger1, d['relation'], trigger2]
        triples.append(triple)

#    print('full graphs: ', triples)

    # extract golden graph in order based on golden explanation
    golden_graph = []
    golden_exp = d['explanation']
    qa_pairs = []
    for exp in golden_exp:
        if '' == exp or 'None' == exp:
            continue
        question, answer = exp.split('?')
        answer = answer.split('@')[0].strip()
        if re.search('#\d', question):
            question = re.sub('#\d', qa_pairs[-1][1], question)
        qa_pairs.append([question, answer])

        top_match = []
        for triple in triples:
            if 3 == len(top_match):
                break
            match = []
            for ele in triple:
                if re.search(ele, question, re.I) or re.search(ele, answer, re.I):
                    match.append(ele)
            if len(match) > len(top_match):
                top_match = match

        golden_graph.append(top_match)

#    print('qa_pairs: ', qa_pairs)
#    print('golden_graph: ', golden_graph)

    # match predicted explanation based on golden graph
    if 'predicted_explanation' in d:
        # for explanation cot
        pred_exp = d['predicted_explanation']
        qa_pairs = []
        for exp in pred_exp:
            if '' == exp:
                continue
            if '?' in exp:
                question, answer = exp.split('?')
                answer = answer.split('@')[0].strip()
                if re.search('#\d', question):
                    question = re.sub('#\d', qa_pairs[-1][1], question)
                qa_pairs.append([question, answer])
            else:
                qa_pairs.append([exp, ''])
    else:
        # for free-form cot
#        pred_exp = d['prediction_ori'].split('. ')[-2:]
        pred_exp = [str(sent).strip() for sent in nlp(d['prediction_ori']).sents][-2:]
        qa_pairs = [[e, ''] for e in pred_exp]

    match_idx = 0
    exp_idx = 0
    match_count = 0
    while exp_idx < len(qa_pairs):
#        print(exp_idx, match_idx)
        question, answer = qa_pairs[exp_idx]
#        print(question, answer)

        flag = False
        for idx, triple in enumerate(golden_graph[match_idx:]):
            match = []
#            print(question, answer, triple)
            for ele in triple:
                if re.search(ele, question, re.I) or re.search(ele, answer, re.I):
                    match.append(ele)
            if 2 <= len(match):
                flag = True
                match_count += 1
                match_idx += idx + 1
                break
        if not flag:
            exp_idx += 1
#        input()

    if 0 == match_count:
        return 0, 0, 0

    precision = match_count / len(pred_exp)
    recall = match_count / len(golden_graph)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall

def cal_incompleteness(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    total_prec = []
    total_recall = []
    total_f1 = []
    for d in tqdm(data):
        f1, prec, recall = incompleteness(d)
        total_prec.append(prec)
        total_recall.append(recall)
        total_f1.append(f1)

        d['incompleteness'] = {
            'precision': prec,
            'recall': recall,
            'f1': f1,
        }
    print(np.sum(total_f1) / len(total_f1))
    print(np.sum(total_prec) / len(total_prec))
    print(np.sum(total_recall) / len(total_recall))

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    file_name = '../results/predictions_cot_golden-event.json'
    cal_consistency(file_name)
#    cal_incompleteness(file_name)

