import copy
import json
import random
import re
import time

import openai
import tiktoken
import yaml
from tqdm import tqdm

with open('src/config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_path = config['train_path']
data = []
with open(data_path, 'r') as f:
    train_data = [json.loads(line) for line in f.readlines()]
    data.extend(train_data)

data_path = config['dev_path']
with open(data_path, 'r') as f:
    dev_data = [json.loads(line) for line in f.readlines()]
    data.extend(dev_data)

data_path = config['test_path']
with open(data_path, 'r') as f:
    test_data = [json.loads(line) for line in f.readlines()]
    data.extend(test_data)

with open(config['openai_key'], 'r') as f:
    openai.api_key = f.readline().strip()

with open(config['openai_org_id'], 'r') as f:
    openai.organization = f.readline().strip()

with open(config['template_path'], 'r') as f:
    templates = json.load(f)

if config['use_doc1_relation']:
    data = data[:1]

#data = data[11:13]

def call(message):
    messages = [{'role': 'user', 'content': message}]

    while True:
        try:
            response = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo-0613',
                            messages=messages,
                            max_tokens=50,
                            temperature=0.2
                            )
            break
        except Exception as e:
            time.sleep(2)
            print('Errrrrrrrrrrrrrrrrrr', str(e))
            print(len(tokenizer.encode(messages[0]['content'])))

    prediction = response['choices'][0]['message']['content']

    return prediction

# relations
relations = ['before', 'after', 'overlaps', 'equals',
             'contains', 'contained by',
             'causes', 'caused by', 'none']

try:
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
except KeyError:
    print("Warning: model not found. Using cl100k_base encoding.")
    tokenizer = tiktoken.get_encoding("cl100k_base")

def event_relation(doc, event1, event2):
#    ### just for offline test ###
#    return 'before'
#    ### --------------------- ###

    message = ''

    e1 = 'Event1:\nTrigger: {t}'.format(t=event1['trigger']['text'])
    for arg in event1['arguments']:
        roles = '{role}: {text}'.format(role=arg['role'], text=arg['text'])
        e1 = '\n'.join([e1, roles])
    message = '\n\n'.join([message, e1])

    e2 = 'Event2:\nTrigger: {t}'.format(t=event1['trigger']['text'])
    for arg in event2['arguments']:
        roles = '{role}: {text}'.format(role=arg['role'], text=arg['text'])
        e2 = '\n'.join([e2, roles])
    message = '\n'.join([message, e2])

    relations_prompt = '\n'.join(['Temporal/Causality Relations:'] + relations)
    message = '\n\n'.join([message, relations_prompt])

    question = 'What is the temporal/causality relation between Event1 and Event2?'
    message = '\n\n'.join([message, question])

    length = len(tokenizer.encode(message)) + 7
    remaining_length = 4097 - 50 - length

    doc_prompt = 'Document:\n{document}'.format(document=doc)
    doc_prompt = tokenizer.decode(tokenizer.encode(doc_prompt)[:remaining_length])
    message = ''.join([doc_prompt, message])

    pred = call(message)
    for r in relations:
        if r in pred:
#            print(json.dumps([message, pred], indent=2), file=f_relations)
#            print(event1['id'], event2['id'], r)
            return r

    return 'none'

def relation_reverse(relation):
    # 'before', 'after', 'overlaps', 'equals', 'contains', 'contained by',
    # 'causes', 'caused by', 'none'])
    if 'before' == relation:
        return 'after'
    if 'after' == relation:
        return 'before'
    if 'contains' == relation:
        return 'contained by'
    if 'contained by' == relation:
        return 'contains'
    if 'causes' == relation:
        return 'caused by'
    if 'cuased by' == relation:
        return 'causes'
    return relation

##### template related #####
def find_template(event):
    event_type = event['event_type']
    ori_type = event_type
    while '' != ori_type:
        try:
            template = templates[event_type]
            break
        except:
            # if cannot find one, replace the first unmatched part with 'Unspecified'
            ori_type = ori_type.split('.')[:-1]
            event_type = ori_type
            while 3 > len(event_type):
                event_type.append('Unspecified')
            event_type = '.'.join(event_type)
    return template

def template_verb_extraction(template):
    verbs = []
    tokens = template['template']
    scopes = template['scopes']
    verbs = tokens[scopes[0]['end']+1:scopes[1]['start']]
    return ' '.join(verbs)

def find_wh_word(template, argument):
    # for the wh-word
    wh = 'what'
    for token in template['scopes']:
        if token['role'] == argument['role']:
            wh = token['wh']
            break

    return wh

def build_partial_question(template, role_options, fill=False):
    init_question = []
    trigger_flag = False
    for idx, token in enumerate(template['scopes']):
        if 0 < idx and not trigger_flag:
            trigger_flag = True
            init_question.append(template['verb'])
        for option in role_options:
            if token['role'] != option[0]:
                continue
            option[0] = ''
            if fill and '' != option[1]:
                text = re.sub('\[.*\]', option[1], token['text'])
                init_question.append(text)
            else:
                init_question.append(token['text'])
            break
    return ' '.join(init_question)

##### event unique check #####
def event_unique(all_events, event):
    '''
    event: {'id': eid(str),
            'event_type': et(str),
            'arguments':[args](structure)}
    '''
    tgt_verb = event['template']['verb']

    flag = True
    for e in all_events:
        if e['id'] == event['id']:
            continue
        e_verb = e['template']['verb']
        if e_verb == tgt_verb:
            cnt = 0
            for tgt_arg in event['arguments']:
                for e_arg in e['arguments']:
                    if tgt_arg['role'] == e_arg['role']:
                        flag = False
                        cnt += 1
                        break
            if len(event['arguments']) == cnt:
                flag = False
        if not flag:
            break

    return flag

##### strategy 1 #####
def relation_text_explicit(relation):
    if 'before' == relation:
        return 'which is before'
    elif 'after' == relation:
        return 'which is after'
    elif 'equals' == relation:
        return 'which equals to'
    elif 'contains' == relation:
        return 'which contains'
    elif 'contained by' == relation:
        return 'which is contained by'
    elif 'overlaps' == relation:
        return 'which overlaps'
    elif 'causes' == relation:
        return 'which causes'
    elif 'caused by' == relation:
        return 'which is caused by'

def template_event_explicit(event1, event2, relation):
    args1 = event1['arguments']
    args2 = event2['arguments']
    qa_pairs = []
    for a1 in args1[:3]:
        for a2 in args2[:3]:
            text = 'What is the {role1} in the event {relation} another event that {argument2} is {role2}?'.format(
                        role1=a1['role'],
                        relation=relation_text_explicit(relation),
                        role2=a2['role'],
                        argument2=a2['text'],
                    )
            qa_pairs.append({'init_question': text,
                             'expect_answer': a1['text'].lower(),
                             'events': [[event1], [event2]],
                             'relation': relation
                             })

    return qa_pairs

def relation_text_implicit(relation):
    if 'before' == relation:
        return 'before'
    elif 'after' == relation:
        return 'after'
    elif 'equals' == relation:
        return 'at the same time'
    elif 'contains' == relation:
        return 'during'
    elif 'contained by' == relation:
        return 'during'
    elif 'overlaps' == relation:
        return 'when'
    elif 'causes' == relation:
        return 'leads to'
    elif 'caused by' == relation:
        return 'because'
    elif 'none' == relation:
        raise Exception

def template_event_implicit(event1, event2, relation, all_events):
    # event 1
    template_1 = event1['template']
    verb1 = event1['template']['verb']
    args1 = event1['arguments']
    if 1 < len(args1):
        args1 = random.sample(args1, k=1)
    event1_trigger = event1['trigger']['text']

    # event 2
    template_2 = event2['template']
    verb2 = event2['template']['verb']
    args2 = event2['arguments']
    if 1 < len(args2):
        args2 = random.sample(args2, k=1)
    event2_trigger = event2['trigger']['text']

    qa_pairs = []
    for idx1, a1 in enumerate(args1):
        # for the wh-word
        wh = find_wh_word(template_1, a1)

        # generate partial question for the first event
        a1_partial_q = build_partial_question(template_1, [[a1['role'], wh]], fill=True)

        # generate partial decomposed QA pairs for explanation
        partial_decomposed_pairs = []
        text = 'What event is {relation} #1 has a {role}?'.format(
                relation=relation,
                role=a1['role'])
        partial_decomposed_pairs.append([text, event1_trigger])
        text = '{pq} in the #2?'.format(
                pq=a1_partial_q
                )
        partial_decomposed_pairs.append([text, a1['text'].lower()])

        # form an event strucutre based on the used argument and test shortcut on it
        # if unique, a shortcut warning prompt shows on the UI for annotators to check
        test_event = copy.deepcopy(event1)
        test_event['arguments'] = [a1]
        shortcut_flag = event_unique(all_events, test_event)

        # save used argument information to the event
        event1_a1 = copy.deepcopy(event1)
        event1_a1['arguments'][idx1]['used'] = True
        event1_a1.pop('template', None)

        for idx2, a2 in enumerate(args2):
            # generate partial question for the second event
            a2_partial_q = build_partial_question(
                                template_2,
                                [[a2['role'], a2['text']]],
                                fill=True)

            template_question = '{pq1} {relation} {pq2}'.format(
                                    pq1=a1_partial_q,
                                    relation=relation_text_implicit(relation),
                                    pq2=a2_partial_q
                                    )

            # generate decomposed pairs for the whole question
            decomposed_pairs = []
            text = 'What event contains {arg} is the {role}?'.format(
                    arg=a2['text'],
                    role=a2['role'])
            decomposed_pairs.append([text, event2_trigger])
            decomposed_pairs.extend(partial_decomposed_pairs)

            # check the unique of start event
            # form an event strucutre based on the used argument and test unique on it
            # if not unique, an prompt shows on the UI for annotators to add information
            test_event = copy.deepcopy(event2)
            test_event['arguments'] = [a2]
            unique_flag = event_unique(all_events, test_event)

            # human rephrase to natural language questions
            if config['human_rephrase']:
                print(json.dumps(event1, indent=2))
                print(json.dumps(event2, indent=2))
                print(relation)
                print(template_question, ' -- ', a1['text'])
                nl_question = input('natural language style question: ')
                if 'x' == nl_question:
                    # delete the question
                    continue
                if '' == nl_question:
                    # input nothing
                    nl_question = text
                elif 1 == len(nl_question.split()):
                    # input is: who, what, which, or so
                    nl_question = '{wh} {verb1} {relation} {partial_q}?'.format(
                                    wh = nl_question,
                                    verb1=verb1,
                                    relation=relation_text_implicit(relation),
                                    partial_q = partial_question
                                    )
                print(nl_question)
            else:
                nl_question = template_question

            # save used argument information to the event
            event2_a2 = copy.deepcopy(event2)
            event2_a2['arguments'][idx2]['used'] = True
            event2_a2.pop('template', None)

            # data format
            qa_pairs.append({'init_question': template_question,
                             'nl_question': nl_question,
                             'expect_answer': [a1['text'].lower()],
                             'events': [[event1_a1], [event2_a2]],
                             'relation': relation,
                             'decompose': decomposed_pairs,
                             'shortcut': shortcut_flag,
                             'locate_start_event': unique_flag,
                             })

    return qa_pairs

#def template_2(doc, eid, event):
#    all_relations = ['before', 'after', 'overlap', 'equal', 'contain']
#    args = event['arguments']
#    cnt1, cnt2 = 0, 0
#    for arg in args:
#        cnt2 += 1
#        for relation in all_relations:
#            text = 'Only output a number: How many events are {relation} the event that {argument} is {role}?'.format(
#                        relation=relation,
#                        argument=arg['text'],
#                        role=arg['role'],
#                    )
#            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc), text])
#            print(message)
#            pred = call(message)
#
#            answer = count_relation(eid, relation)
#            if str(answer) in pred:
#                cnt1 += 1
#            print(json.dumps([text, pred], indent=2), file=f_outputs2)
#            print(event['id'], pred)
#
#    return cnt1, cnt2
#
#def template_3(doc, event):
#    relations = ['before', 'after', 'overlap', 'equal', 'contains']
#    args = event['arguments']
#    cnt1, cnt2 = 0, 0
#    for arg in args:
#        cnt2 += 1
#        for relation in relations:
#            text = 'Question: List all events are {relation} the event that {argument} is {role}'.format(
#                        relation=relation,
#                        argument=arg['text'],
#                        role=arg['role'],
#                    )
#            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc), text])
#            pred = call(message)
#            print(json.dumps([text, pred], indent=2), file=f_outputs3)
#            print(event['id'], pred)
#
#    return cnt1, cnt2

##### strategy 2 #####
event_histories = set([])

def find_next(text_history, event_history, entity_event_dict, hop=-1):
#    print(json.dumps(event_history, indent=2))
#    print(json.dumps(text_history))
#    input()
    if hop >= config['max_events']:
        event_histories.add(tuple([e['id'] for e in event_history] + text_history[:-1]))
        return

    event_list = entity_event_dict[text_history[-1]]
    for event in event_list:
        flag = False
        for h in event_history:
            if event['id'] == h['id']:
                flag = True
                break
        if flag:
            continue

        event_history.append(event)
        for arg in event['arguments']:
            flag = False
            for h in text_history:
                if arg['text'] == h:
                    flag = True
            if flag and (hop + 1 != config['max_events']):
                continue

            text_history.append(arg['text'])
            find_next(text_history, event_history, entity_event_dict, hop + 1)
            text_history.pop()
        event_history.pop()

def template_entity_nhop_implicit(doc, events, text_history, all_events):
    partial_questions = [] # record argument role description
    decomposed_pairs = [] # record explanations
    shortcut_event_idxes = [] # record possible shortcut event indexes
    annotated_events = [] # add used argument information to events

    # the first event
    # find template
    template = events[0]['template']
    # find corresponding argument given bridging text
    for arg in events[0]['arguments']:
        if arg['text'] == text_history[0]:
            bridging_arg = arg
            break
    # pick an answer in the first event
    options = []
    for arg in events[0]['arguments']:
        if arg['text'] != text_history[0]: # and arg['role'] != bridging_arg['role']:
            options.append(arg)
    if 0 == len(options):
        return None
    answer = random.choice(options)
    wh_word = find_wh_word(template, answer) # answer wh word
    # build a partial question
    role_options = [[answer['role'], wh_word],
                    [bridging_arg['role'], '']]
    partial_question = build_partial_question(template, role_options, fill=True)
    partial_questions.append(partial_question)
    # build an explanation step
    role_options = [[answer['role'], wh_word],
                    [bridging_arg['role'], '#{0}'.format(len(text_history))]]
    explanation = build_partial_question(template, role_options, fill=True)
    decomposed_pairs.append([explanation, answer['text'].lower()])
    # form an event strucutre based on the used argument and test shortcut on it
    # if unique, a shortcut warning prompt shows on the UI for annotators to check
    test_event = copy.deepcopy(events[0])
    test_event['arguments'] = [answer, bridging_arg]
    shortcut_flag = event_unique(all_events, test_event)
    if shortcut_flag:
        shortcut_event_idxes.append(1)
    # save used argument information to the event
    event_1_args = copy.deepcopy(events[0])
    for arg in [answer, bridging_arg]:
        for e_arg in event_1_args['arguments']:
            if arg['entity_id'] == e_arg['entity_id']:
                e_arg['used'] = True
    event_1_args.pop('template', None)
    annotated_events.append(event_1_args)

    # middle events if exist
    for i in range(1, len(text_history)):
        # find template
        template = events[i]['template']
        # find corresponding arguments given bridging texts
        bridging_args = []
        for text in text_history[i-1:i+1]:
            for arg in events[i]['arguments']:
                if arg['text'] == text:
                    bridging_args.append(arg)
        wh_word = find_wh_word(template, bridging_args[0]) # the first bridge's wh word
        # build a partial question
        role_options = [[bridging_args[0]['role'], wh_word],
                        [bridging_args[1]['role'], '']]
        partial_question = build_partial_question(template, role_options, fill=True)
        partial_questions.append(partial_question)
        # build an explanation step
        role_options = [[bridging_args[0]['role'], wh_word],
                        [bridging_args[1]['role'], '#{0}'.format(len(text_history)-i)]]
        explanation = build_partial_question(template, role_options, fill=True)
        decomposed_pairs.append([explanation, bridging_args[0]['text']])
        # form an event strucutre based on the used argument and test shortcut on it
        # if unique, a shortcut warning prompt shows on the UI for annotators to check
        test_event = copy.deepcopy(events[i])
        test_event['arguments'] = bridging_args
        shortcut_flag = event_unique(all_events, test_event)
        if shortcut_flag:
            shortcut_event_idxes.append(i+1)
        # save used argument information to the event
        event_i_args = copy.deepcopy(events[i])
        for arg in bridging_args:
            for e_arg in event_i_args['arguments']:
                if arg['entity_id'] == e_arg['entity_id']:
                    e_arg['used'] = True
        event_i_args.pop('template', None)
        annotated_events.append(event_i_args)

    # the last event
    # find template
    template = events[-1]['template']
    # find corresponding argument given bridging text
    for arg in events[-1]['arguments']:
        if arg['text'] == text_history[-1]:
            bridging_arg = arg
            break
    wh_word = find_wh_word(template, bridging_arg) # last bridge's wh word
    # build a partial question
    role_options = [[bridging_arg['role'], wh_word]]
    partial_question = build_partial_question(template, role_options, fill=True)
    partial_questions.append(partial_question)
    # build an explanation step
    role_options = [[bridging_arg['role'], wh_word]]
    explanation = build_partial_question(template, role_options, fill=True)
    decomposed_pairs.append([explanation, bridging_arg['text']])
    # check the unique of start event
    # form an event strucutre based on the used argument and test unique on it
    # if not unique, an prompt shows on the UI for annotators to add information
    test_event = copy.deepcopy(events[-1])
    test_event['arguments'] = [bridging_arg]
    unique_flag = event_unique(all_events, test_event)
    # save used argument information to the event
    event_last_args = copy.deepcopy(events[-1])
    for e_arg in event_last_args['arguments']:
        if bridging_arg['entity_id'] == e_arg['entity_id']:
            e_arg['used'] = True
    event_last_args.pop('template', None)
    annotated_events.append(event_last_args)

    # join all argument role descriptions and explanations
    template_question = ' '.join(partial_questions)
    decomposed_pairs = decomposed_pairs[::-1]

    # human rephrase to natural language in console
    if config['human_rephrase']:
        print(json.dumps(events, indent=2))
        print(answer, ' -- ', text_history)
        nl_question = input('natural language style question: ')
        if 'x' == nl_question:
            return None
        if '' == nl_question:
            nl_question = template_question
        print(nl_question)
    else:
        nl_question = template_question

    # data format
    qa_pair = {'init_question': template_question,
               'nl_question': nl_question,
               'expect_answer': [answer['text'].lower()],
               'events': annotated_events,
               'bridge_entities': text_history,
               'decompose': decomposed_pairs,
               'shortcut': shortcut_event_idxes,
               'locate_start_event': unique_flag,
               }

    return qa_pair

##### strategy 3 #####
def count_relation(event, relation, relation_pairs):
    cnt = 0
    events = []
    for (e1, r, e2) in relation_pairs:
        if e1['id'] == event['id'] and r == relation:
            cnt += 1
            events.append(e2)
    return cnt, events

def relation_text_strategy_3(relation):
    if 'before' == relation:
        return 'before'
    elif 'after' == relation:
        return 'after'
    elif 'equal' == relation:
        return 'at the same time as'
    elif 'contain' == relation:
        return 'during'
    elif 'overlap' == relation:
        return 'when'
    elif 'none' == relation:
        raise Exception

def template_event_list_count(doc, events, relation_pairs=None):
    qa_pairs = []

    event_types = {}
    for event in events:
        event_type = event['event_type']
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(event)
    for k, v in event_types.items():
        if len(qa_pairs) > 2:
            break
        template_1 = 'How many events are related to {event_type}'.format(event_type=k)
        decomposed_pairs = [['What events are related to {event_type}'.format(event_type=k),
                                [e['trigger']['text'] for e in v]],
                            ['How many of #1 in total?', len(v)]]
        qa_pairs.append({'init_question': template_1,
                         'nl_question': '',
                         'expect_answer': len(v),
                         'events': v,
                         'decompose': decomposed_pairs,
                         })

    roles = {}
    for event in events:
        for arg in event['arguments']:
            key = '||'.join([arg['role'], arg['text']])
            if arg['role'] == '' or arg['text'] == '' or len(key.split('||')) != 2:
                print(key, json.dumps(event))
                input()
            if key not in roles:
                roles[key] = []
            roles[key].append(event)
    for k, v in roles.items():
        if len(qa_pairs) > 5:
            break
        role, arg = k.split('||')
        template_2 = 'How man times is the {argument} mentioned as the {role}'.format(
                        argument=arg,
                        role=role,
                        )
        decomposed_pairs = [['What events contain the {argument} as the {role}'.format(
                                argument=arg,
                                role=role,
                                ),
                                [e['trigger']['text'] for e in v]],
                            ['How many of #1 in total?', len(v)]]
        qa_pairs.append({'init_question': template_2,
                         'nl_question': '',
                         'expect_answer': len(v),
                         'events': v,
                         'decompose': decomposed_pairs,
                         })

    roles = {}
    for event in events:
        for arg in event['arguments']:
            if arg['role'] not in roles:
                roles[arg['role']] = []
            roles[arg['role']].append(event)
    for k, v in roles.items():
        if len(qa_pairs) > 8:
            break
        template_3 = 'How many {role} in the whole text'.format(role=k)
        decomposed_pairs = [['What are {role} in the whole text?',
                                [e['trigger']['text'] for e in v]],
                            ['How many of #1 in total?', len(v)]]
        qa_pairs.append({'init_question': template_3,
                         'nl_question': '',
                         'expect_answer': len(v),
                         'events': v,
                         'decompose': decomposed_pairs,
                         })

#    for arg in args:
#        for relation in relations:
#            text = 'How many events happen {relation} the event that {argument} is {role}'.format(
#                        relation=relation_text_strategy_3(relation),
#                        argument=arg['text'],
#                        role=arg['role'],
#                    )
#            '''
#            1. xxx
#            2. xxx
#            There are [number] events [relation] the event xxx.
#            '''
#            answer, events_list = count_relation(event, relation, relation_pairs)
#            qa_pairs.append({'init_question': text,
#                             'expect_answer': answer,
#                             'event': event,
#                             'relation': relation,
#                             'events': events_list})

    return qa_pairs

def strategy_1(doc, events, relation_pairs):
    questions_s1 = []
    for (e1, r, e2) in relation_pairs:
        qa_pairs = template_event_implicit(e1, e2, r, events)
        questions_s1.extend(qa_pairs)
 
    # human rephrase
#    save_list = []
#    for q in questions_s1:
#        if 10 <= len(save_list):
#            break
#        print(json.dumps(q, indent=2))
#        ox = input(q['init_question'] + '\n')
#        if 'x' == ox:
#            continue
#        save_list.append(q)
#    with open('s1_test_im.json', 'w') as f:
#        json.dump(save_list, f, indent=2)

    if config['prediction']:
        cnt = 0
        for qa_pairs in questions_s1:
            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc),
                                   qa_pairs['nl_question']])
#            pred = call(message)
#            qa_pairs['prediction'] = pred
#            print(qa_pairs['nl_question'], ' -- ',
#                  pred, ' -- ',
#                  qa_pairs['expect_answer'])
            flag = True
            for answer in qa_pairs['expect_answer']:
                if answer not in pred.lower():
                    flag = False
                    break
            if flag:
                cnt += 1
#            if config['debug']:
#                break
        if len(questions_s1):
            print(cnt / len(questions_s1))
#        with open(config['strategy_1_data'], 'w') as f:
#            json.dump(questions_s1, f, indent=2)

    # add an empty question case for additional questions
    questions_s1.append({'init_question': 'additional question',
                         'nl_question': 'additional question',
                         'expect_answer': [''],
                         'events': [[], []],
                         'relation': 'none',
                         'decompose': [[], [], []],
                         'shortcut': False,
                         'locate_start_event': True,
                         })

    return questions_s1
    
def strategy_2(doc, events):
    # build entity bridging dict
    entity_event_dict = {}
    for event in events:
        arguments = event['arguments']
        if 2 > len(arguments):
            continue
        for arg in arguments:
            if arg['text'] not in entity_event_dict:
                entity_event_dict[arg['text']] = []
            entity_event_dict[arg['text']].append(event)

    # find lists
    questions_s2 = []
    for hop in range(1, 4):
        global event_histories
        event_histories = set([])
        config['max_events'] = hop
        for event in events:
            arguments = event['arguments']
            if 2 > len(arguments):
                continue
            else:
                roles = set([arg['role'] for arg in event['arguments']])
                if 2 > len(roles):
                    continue
            for arg in arguments:
                event_history = [event]
                text_history = [arg['text']]
                find_next(text_history, event_history, entity_event_dict, 0)

#        print(event_histories)
        if config['max_q_per_doc_hop'] < len(event_histories):
            event_histories = random.sample(event_histories,
                                            k=config['max_q_per_doc_hop'])
        print(hop, len(event_histories))

        for histories in event_histories:
            event_list = []
            for i in range(config['max_events']+1):
                for event in events:
                    if histories[i] == event['id']:
                        event_list.append(event)
                        break
            event_list = event_list[::-1]
            text_history = histories[config['max_events']+1:][::-1]
            qa_pair = template_entity_nhop_implicit(doc, event_list, text_history, events)
            if qa_pair is None:
                continue
            questions_s2.append(qa_pair)

    # human rephrase
#    save_list = []
#    for q in questions_s2:
#        if 10 <= len(save_list):
#            break
#        print(json.dumps(q, indent=2))
#        ox = input(q['init_question'] + '\n')
#        if 'x' == ox:
#            continue
#        save_list.append(q)
#    with open('s2_test_im.json', 'w') as f:
#        json.dump(save_list, f, indent=2)

#    questions_s2 = questions_s2[:10]
    if config['s2_cot']:
        with open(config['s2_example'], 'r') as f:
            example = json.load(f)
            qa_pair = '\n'.join(['Question: {q}'.format(q=example['question']),
                                 'Answer: {a}'.format(a=example['reasoning_chain'])])
            example = '\n\n'.join(['Document:\n{doc}'.format(doc=example['doc']),
                                   qa_pair])

    if config['prediction']:
        cnt = 0
        for qa_pairs in tqdm(questions_s2):
            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc),
                                   qa_pairs['nl_question']])
            if config['s2_cot']:
                message = '\n\n'.join([example, message])
            pred = call(message)
            qa_pairs['prediction'] = pred
#            print(qa_pairs['nl_question'], ' -- ',
#                  pred, ' -- ',
#                  qa_pairs['expect_answer'])
            flag = True
            for answer in qa_pairs['expect_answer']:
                if answer not in pred.lower():
                    flag = False
                    break
            if flag:
                cnt += 1
#            if config['debug']:
#                break
        if len(questions_s2):
            print(cnt / len(questions_s2))
#        with open(config['strategy_2_data'], 'w') as f:
#            json.dump(questions_s2, f)

    questions_s2.append({'init_question': 'additional question',
                         'nl_question': 'additional question',
                         'expect_answer': [''],
                         'events': [],
                         'bridge_entities': ['', ''],
                         'decompose': [[], [], []],
                         'shortcut': [],
                         'locate_start_event': True,
                         })

    return questions_s2

def strategy_3(doc, events, relation_pairs):
    questions_s3 = []
    questions_s3 = (template_event_list_count(doc, events))
#    for event in events:
#        qa_pairs = template_event_list_count(doc, event, relation_pairs)
#        questions_s3.extend(qa_pairs)

    # human rephrase
#    save_list = []
#    for q in questions_s3:
#        if 10 <= len(save_list):
#            break
#        print(json.dumps(q, indent=2))
#        ox = input(q['init_question'] + '\n')
#        if 'x' == ox:
#            continue
#        save_list.append(q)
#    with open('s3_test.json', 'w') as f:
#        json.dump(save_list, f, indent=2)

#    if config['prediction']:
#        questions_s3 = questions_s3[:10]
#        d = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
#              6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
#              11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
#              15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
#              19 : 'nineteen', 20 : 'twenty'}
#        cnt = 0
#        for qa_pairs in questions_s3:
#            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc),
#                                   qa_pairs['init_question']])
#            pred = call(message)
#            qa_pairs['prediction'] = pred
##            print(qa_pairs['init_question'], ' -- ',
##                  pred, ' -- ',
##                  qa_pairs['expect_answer'])
#            expect_answer = qa_pairs['expect_answer']
#            if ((str(expect_answer) in pred.lower()) or
#                    (isinstance(expect_answer, int) and
#                        d[expect_answer] in pred.lower())):
#                cnt += 1
#        print(cnt / len(questions_s3))
##        with open(config['strategy_3_data'], 'w') as f:
##            json.dump(questions_s3, f)

#    questions_s3.append({'init_question': 'additional question',
#                         'nl_question': 'additional question',
#                         'expect_answer': [''],
#                         'events': [],
#                         'decompose': [[], [], []],
#                         'shortcut': False,
#                         'locate_start_event': True,
#                         })

    return questions_s3

def strategy_4(doc, events):
#    event_types_set = set([event['event_type'] for event in events])
#    event_types = {}
#    for event_type in event_types_set:
#        event_list = []
#        for event in events:
#            if event_type == event['event_type']:
#                event_list.append(event)
#        if 1 < len(event_list):
#            event_types[event_type] = event_list
#
#    # human annotator required
#    questions_s4 = []
#    for k, v in event_types.items():
#        print(json.dumps(v, indent=2))
#        print(k)
#        text = input('Question: ')
#        if 'x' == text:
#            continue
#        elif 'b' == text:
#            break
#        if config['prediction']:
#            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc),
#                                   text])
#            pred = call(message)
#        answer = input('Answer: ')
#        questions_s4.append({'init_question': '',
#                             'nl_question': '',
#                             'expect_answer': '',
#                             'events': v,
#                             'decompose': [[], []]
#                             })

    qa_pairs = []

    event_types = {}
    for event in events:
        event_type = event['event_type']
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(event)
    for k, v in event_types.items():
        if len(qa_pairs) > 2:
            break
        template_1 = 'How many events are related to {event_type}'.format(event_type=k)
        decomposed_pairs = [['What events are related to {event_type}'.format(event_type=k),
                                [e['trigger']['text'] for e in v]],
                            ['How many of #1 in total?', len(v)]]
        qa_pairs.append({'init_question': template_1,
                         'nl_question': '',
                         'expect_answer': len(v),
                         'events': v,
                         'decompose': decomposed_pairs,
                         })

    roles = {}
    for event in events:
        for arg in event['arguments']:
            key = '||'.join([arg['role'], arg['text']])
            if arg['role'] == '' or arg['text'] == '' or len(key.split('||')) != 2:
                print(key, json.dumps(event))
                input()
            if key not in roles:
                roles[key] = []
            roles[key].append(event)
    for k, v in roles.items():
        if len(qa_pairs) > 5:
            break
        role, arg = k.split('||')
        template_2 = 'How man times is the {argument} mentioned as the {role}'.format(
                        argument=arg,
                        role=role,
                        )
        decomposed_pairs = [['What events contain the {argument} as the {role}'.format(
                                argument=arg,
                                role=role,
                                ),
                                [e['trigger']['text'] for e in v]],
                            ['How many of #1 in total?', len(v)]]
        qa_pairs.append({'init_question': template_2,
                         'nl_question': '',
                         'expect_answer': len(v),
                         'events': v,
                         'decompose': decomposed_pairs,
                         })

    roles = {}
    for event in events:
        for arg in event['arguments']:
            if arg['role'] not in roles:
                roles[arg['role']] = []
            roles[arg['role']].append(event)
    for k, v in roles.items():
        if len(qa_pairs) > 8:
            break
        template_3 = 'How many {role} in the whole text'.format(role=k)
        decomposed_pairs = [['What are {role} in the whole text?'.format(role=k),
                                [e['trigger']['text'] for e in v]],
                            ['How many of #1 in total?', len(v)]]
        qa_pairs.append({'init_question': template_3,
                         'nl_question': '',
                         'expect_answer': len(v),
                         'events': v,
                         'decompose': decomposed_pairs,
                         })

    event_types = {}
    for event in events:
        event_type = event['event_type']
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(event)

    for k, v in event_types.items():
        if len(qa_pairs) > 11:
            break
        qa_pairs.append({'init_question': '',
                         'nl_question': '',
                         'expect_answer': '',
                         'events': v,
                         'decompose': [[], []]
                         })

    triggers = {}
    for event in events:
        trigger = event['trigger']['text']
        if trigger not in triggers:
            triggers[trigger] = []
        triggers[trigger].append(event)

    for k, v in triggers.items():
        if len(qa_pairs) > 14:
            break
        qa_pairs.append({'init_question': '',
                         'nl_question': '',
                         'expect_answer': '',
                         'events': v,
                         'decompose': [[], []]
                         })

    return qa_pairs

# for strategy 1
def combine_questions(questions):
    '''
    'init_question': text,
    'nl_question': nl_text,
    'expect_answer': [a1['text'].lower()],
    'events': [event1, event2],
    'relation': relation})
    '''
    unique_questions = {}
    for question in questions:
        nl_q = question['nl_question']
        if nl_q not in unique_questions:
            unique_questions[nl_q] = question
        else:
            unique_questions[nl_q]['expect_answer'].extend(question['expect_answer'])

            # extend the first event list
            flag = False
            for event in unique_questions[nl_q]['events'][0]:
                if question['events'][0][0]['id'] == event['id']:
                    flag = True
            if not flag:
                unique_questions[nl_q]['events'][0].extend(question['events'][0])

            # extend the second event list
            flag = False
            for event in unique_questions[nl_q]['events'][1]:
                if question['events'][1][0]['id'] == event['id']:
                    flag = True
            if not flag:
                unique_questions[nl_q]['events'][1].extend(question['events'][1])

    return list(unique_questions.values())

questions = {'strategy_1': {},
             'strategy_2': {},
             'strategy_3': {},
             'strategy_4': {},
             }
tcnt0, tcnt1, tcnt2 = 0, 0, 0
for d in tqdm(data):
    doc = d['text']
#    events = d['event_mentions']
    events = sorted(d['event_mentions'], key=lambda x: x['trigger']['start'])
    for event in events:
        template = find_template(event)
        verb = template_verb_extraction(template)
        event['template'] = template
        event['template']['verb'] = verb

    # if only generate for s2, we don't need relations anymore
    if config['generate_relation']:
#    if not ('s2' in config['strategies'] and 1 == len(config['strategies'])):
        # extract relations
        if not config['use_doc1_relation']:
            pair_list = []
            for i in range(len(events)):
                for j in range(i, len(events)):
                    pair_list.append((i, j))
            random.shuffle(pair_list)

            # get one from each relation
            relation_pairs = []
            for pid, pair in enumerate(pair_list):
                if config['max_r_pair_per_doc'] <= len(relation_pairs) or 20 <= pid:
                    break
                i, j = pair
                relation = event_relation(doc, events[i], events[j])
                if 'none' != relation:
                    relation_pairs.append([events[i], relation, events[j]])

            #pairs_dict = {}
            #for i in range(len(events)):
            #    for j in range(i, len(events)):
            #        relation = event_relation(doc, events[i], events[j])
            #        if 'none' != relation:
            #            pairs.append([events[i], relation, events[j]])
        else:
            # only for the first dev document, golden relations for testing performance
            with open(config['doc1_relations'], 'r') as f:
                pairs = [line.split() for line in f.readlines()]
            pairs = [[events[int(i)-1], relation, events[int(j)-1]] for (i, relation, j) in pairs]

            # relation in both directions
            relation_pairs = pairs
#            double_pairs = []
#            for (e1, r, e2) in pairs:
#                double_pairs.append([e1, r, e2])
#                reverse_r = relation_reverse(r)
#                double_pairs.append([e2, reverse_r, e1])
#            relation_pairs = double_pairs
            relation_pairs = random.sample(relation_pairs,
                                           k=config['max_r_pair_per_doc'])

#        if config['debug']:
#            relation_pairs = relation_pairs[:1]
        with open(config['relation_pairs'], 'a') as f:
            json.dump(relation_pairs, f)
            f.write('\n')

    # strategy 1 event relations
    if 's1' in config['strategies']:
        questions_s1 = strategy_1(doc, events, relation_pairs)
        questions_s1 = combine_questions(questions_s1)
        questions['strategy_1'][d['doc_id']] = questions_s1

    # strategy 2 entity bridging
    if 's2' in config['strategies']:
        questions_s2 = strategy_2(doc, events)
        questions['strategy_2'][d['doc_id']] = questions_s2

    # strategy 3 listing and counting
    if 's3' in config['strategies']:
        relation_pairs = []
        questions_s3 = strategy_3(doc, events, relation_pairs)
        questions['strategy_3'][d['doc_id']] = questions_s3

    # strategy 4 comparison
    if 's4' in config['strategies']:
        questions_s4 = strategy_4(doc, events)
        questions['strategy_4'][d['doc_id']] = questions_s4

with open(config['strategy_all_data'], 'w') as f:
    json.dump(questions, f, indent=2)

def convert_format(q, name):
    d = {}
    for k, v in q.items():
        d[k] = {name: v}
    return d

if 's1' in config['strategies']:
    cnt = 0
    for k, v in questions['strategy_1'].items():
        cnt += len(v) - 1
    print(cnt)
    with open(config['strategy_1_data'], 'w') as f:
        json.dump(convert_format(questions['strategy_1'], 'strategy_1'), f, indent=2)

if 's2' in config['strategies']:
    cnt = 0
    for k, v in questions['strategy_2'].items():
        cnt += len(v) - 1
    print(cnt)
    with open(config['strategy_2_data'], 'w') as f:
        json.dump(convert_format(questions['strategy_2'], 'strategy_2'), f, indent=2)

if 's3' in config['strategies']:
    with open(config['strategy_3_data'], 'w') as f:
        json.dump(convert_format(questions['strategy_3'], 'strategy_3'), f, indent=2)

if 's4' in config['strategies']:
    with open(config['strategy_4_data'], 'w') as f:
        json.dump(convert_format(questions['strategy_4'], 'strategy_4'), f, indent=2)

