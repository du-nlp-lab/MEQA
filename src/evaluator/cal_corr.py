import json
import pandas as pd
from functools import reduce

human = pd.read_csv('../results/incompleteness_inconsistency.csv')
print(human.columns)

#avoid_list = [4, 9, 10, 12, 14]
avoid_list = []

#human_ids = human['Example ID'].tolist()
idxes = human['Index'].tolist()
new_idxes = [idxes[i] for i in range(len(idxes)) if i not in avoid_list]
idxes = new_idxes
print(idxes)

suffixes = ['',  '_golden-entity',  '_golden-trigger', '_golden-trigger-arguments', '_graph']
#for suffix in suffixes:
#    print('exp' + suffix)
#    with open(f'../results/predictions_exp{suffix}.json', 'r') as f:
#        data = json.load(f)
#    
#    extracted_data = [data[i] for i in idxes]
#    
#    human_labels = human[f'Incompleteness ({suffix[1:]})'].tolist()
#    human_labels = [human_labels[i] for i in range(len(human_labels)) if i not in avoid_list]
#    print(human_labels)
#    human_labels = pd.Series(human_labels)
#    
#    pred_labels = []
#    for d in extracted_data:
#        incompleteness = int(d['incompleteness']['recall'] * len(d['explanation']))
#        pred_labels.append(incompleteness)
#    print(pred_labels)
#    pred_labels = pd.Series(pred_labels)
#    
#    print(human_labels.corr(pred_labels))
#    print(human_labels.corr(pred_labels, method='spearman'))
#    print(human_labels.corr(pred_labels, method='kendall'))

for suffix in suffixes:
    with open(f'../results/predictions_exp{suffix}.json', 'r') as f:
        data = json.load(f)
    
    extracted_data = [data[i] for i in idxes]
    
    human_labels = human[f'Inconsistency ({suffix[1:]})'].tolist()
    human_labels = [eval(human_labels[i]) for i in range(len(human_labels)) if i not in avoid_list]
    human_labels = reduce(lambda x, y: x + y, human_labels)
    print(human_labels)
    human_labels = pd.Series(human_labels)

    pred_labels = []
    for d in extracted_data:
        consistency = d['pred_consistency']
        pred_labels.append(consistency)
    pred_labels = reduce(lambda x, y: x + y, pred_labels)
    print(pred_labels)
    pred_labels = pd.Series(pred_labels)
 
    print(human_labels.corr(pred_labels))
    print(human_labels.corr(pred_labels, method='spearman'))
    print(human_labels.corr(pred_labels, method='kendall'))

