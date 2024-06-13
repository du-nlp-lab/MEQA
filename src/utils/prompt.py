import json
import re

import utils.openai_api as openai_api

def generate_entity_graph_prompt(d):
    document = '\n'.join(['Document:', d['context']])
    events = d['events']
    entity_list = []
    for event in events:
        for argument in event['arguments']:
            entity_list.append(argument['text'])
    entity_list = list(set(entity_list))
    entities = '\n'.join(entity_list)
    entities = '\n'.join(['Entities:', entities])
    graph_prompt = 'Given the document, please output shorten relations only between above entities in (entity, relation, entity) format:'
    if 'entity_graph' in d:
        graph = '\n'.join([graph_prompt, d['entity_graph']])
    else:
        graph = graph_prompt
#        graph = 'Graph:\n'

    prompt = '\n\n'.join([document, entities, graph])
    return prompt

def compose_prompt(d, tokenizer=None, example=None, cot=False, exp=False, posthoc=False, graph=''):
    total_tokens = 512 # 4097 - 100 - 7 # 100 from output, 7 from openai
    message = ''

    if not cot:
        if not exp:
            # plain qa
            question = '\n'.join(['Please answer the question:', d['question'], 'Answer:'])
        else:
            if not posthoc:
                # cot
                question = '\n'.join(['Please decompose and answer the question:', d['question'], 'Answer:'])
            else:
                # posthoc
                question = '\n'.join(['Please answer the question and then explain the answer:', d['question'], 'Answer:'])
    else:
        # free-form cot
        question = '\n'.join(['Please answer the question step-by-step:', d['question'], 'Answer:'])
    if tokenizer:
        question_tokens = tokenizer.encode(question)

    if example:
        message = example
        example_tokens = tokenizer.encode(example)
        total_tokens -= len(example_tokens) + 2 # 2 from \n\n

    document = '\n'.join(['Document:', d['context']])

    if 'golden-event' == graph:
        events = d['events']
        triples = []
        # connect all arguments and events
        for event in events:
            trigger = event['trigger']['text']
            for argument in event['arguments']:
                triple = '({trigger}, {rel}, {arg})'.format(
                                trigger=trigger,
                                rel=argument['role'],
                                arg=argument['text']
                            )
                triples.append(triple)
        # for strategy 1, connect events by event relations
        if 'relation' in d:
            trigger1 = events[0]['trigger']['text']
            trigger2 = events[1]['trigger']['text']
            triple = '({trigger1}, {rel}, {trigger2})'.format(
                            trigger1=trigger1,
                            rel=d['relation'],
                            trigger2=trigger2
                        )
            triples.append(triple)
        triples = '\n'.join(triples)
        triples = '\n'.join(['Graph:', triples])
        if "" == message:
            message = '\n\n'.join([document, triples, question])
        else:
            message = '\n\n'.join([message, document, triples, question])
    elif 'golden-trigger-arguments' == graph:
        events = d['events']
        triples = []
        # connect all arguments and events
        for event in events:
            trigger = event['trigger']['text']
            for argument in event['arguments']:
                triple = '({trigger}, {arg})'.format(
                                trigger=trigger,
                                arg=argument['text']
                            )
                triples.append(triple)
        # for strategy 1, connect events by event relations
        if 'relation' in d:
            trigger1 = events[0]['trigger']['text']
            trigger2 = events[1]['trigger']['text']
            triple = '({trigger1}, {trigger2})'.format(
                            trigger1=trigger1,
                            trigger2=trigger2
                        )
            triples.append(triple)
        triples = '\n'.join(triples)
        triples = '\n'.join(['Graph:', triples])
        if "" == message:
            message = '\n\n'.join([document, triples, question])
        else:
            message = '\n\n'.join([message, document, triples, question])
    elif 'predict-entity-graph' == graph:
        if 'entity_graph' not in d:
            with open('../data/meqa/example.json', 'r') as f:
                example = json.load(f)

            example_graph_prompt = generate_entity_graph_prompt(example)
            graph_prompt = generate_entity_graph_prompt(d)
            prompt = '\n\n'.join([example_graph_prompt, graph_prompt])
#            print(prompt)
            graph = openai_api.call(prompt, max_tokens=200)
#            print(graph)
#            input()
            d['entity_graph'] = graph
            graph = '\n'.join(['Graph:', graph])
        else:
            graph = '\n'.join(['Graph:', d['entity_graph']])
        if "" == message:
            message = '\n\n'.join([document, graph, question])
        else:
            message = '\n\n'.join([message, document, graph, question])
    elif 'golden-entity' == graph:
        events = d['events']
        entity_list = []
        for event in events:
            for argument in event['arguments']:
                entity_list.append(argument['text'])
        entity_list = list(set(entity_list))
        entities = ', '.join(entity_list)
        entities = '\n'.join(['Entities:', entities])
        if "" == message:
            message = '\n\n'.join([document, entities, question])
        else:
            message = '\n\n'.join([message, document, entities, question])
    elif 'golden-trigger' == graph:
        events = d['events']
        trigger_list = []
        for event in events:
            trigger = event['trigger']['text']
            trigger_list.append(trigger)
        triggers = ', '.join(trigger_list)
        triggers = '\n'.join(['Triggers:', triggers])
        if "" == message:
            message = '\n\n'.join([document, triggers, question])
        else:
            message = '\n\n'.join([message, document, triggers, question])
    elif '' == graph:
#        if tokenizer:
#            document_tokens = tokenizer.encode(document)
#            document_token_len = total_tokens - 2 - len(question_tokens) # 2 from \n\n
#            document = tokenizer.decode(document_tokens[:document_token_len])
        if "" == message:
            message = '\n\n'.join([document, question])
        else:
            message = '\n\n'.join([message, document, question])

    return message

def compose_example(d, cot=False, exp=False, posthoc=False, graph=''):
    message = compose_prompt(d, cot=cot, exp=exp, posthoc=posthoc, graph=graph)
    answer = compose_answer(d, cot=cot, exp=exp, posthoc=posthoc)

    message = '\n'.join([message, answer])
    return message

def compose_answer(d, cot=False, exp=False, posthoc=False):
    answer = d['answer']
    explanation = ''
    for idx, explain in enumerate(d['explanation']):
        explain = re.sub('@\d+', '', explain)
        explanation += str(idx+1) + '. ' + explain + '\n'
    # add the answer if the explanation doesn't have one
    if '?' == explanation.strip()[-1]:
        explanation += ' ' + d['answer']

    if not cot:
        if not exp:
            # plain qa A
            answer = d['answer']
        else:
            if not posthoc:
                # cot E+A
                answer = explanation
                answer += 'So, the answer is: ' + d['answer'] + '.'
            else:
                # post-hoc A+E
                explanation = '\n'.join(['Explanation:', explanation])
                answer = '\n'.join([d['answer'], explanation])
    else:
        # free-form cot FE+A
        answer = d['reasoning_chain']
    return answer

def extract_answer(prediction, cot=False, exp=False, posthoc=False):
    prediction_ori = prediction
    if not cot:
        # extract from pattern "the answer is:"
        extract = re.search('the answer is(.*)$', prediction, re.IGNORECASE)
        if extract is not None:
            prediction = extract.group(1)
            prediction = re.sub(r'^\W+|\W+$', '', prediction)

        if exp:
            if not posthoc:
                # cot E+A
                if prediction == prediction_ori:
                    explanation = prediction.split('\n')
                    # extract from explanation pattern
                    extract = re.search('(.*)?([^?]+)$', prediction, re.IGNORECASE)
                    if extract is not None:
                        prediction = extract.group(2)
                        prediction = re.sub(r'^\W+|\W+$', '', prediction)
                else:
                    explanation = prediction_ori.split('\n')[:-1]
                return prediction, explanation
            else:
                # Post-hoc A+E
                # extract answer sentence
                extract = re.search('((.|\n)*)Explanation:((.|\n)*)$', prediction, re.IGNORECASE)
                explanation = []
                if extract is not None:
                    prediction = extract.group(1)
                    prediction = re.sub(r'^\W+|\W+$', '', prediction)
                    explanation = extract.group(3).split('\n')
                    return prediction, explanation
    else:
        # free-form cot FE+A
        # extract from pattern "the answer is:"
        extract = re.search('the answer is(.*)$', prediction, re.IGNORECASE)
        if extract is not None:
            prediction = extract.group(1)
            prediction = re.sub(r'^\W+|\W+$', '', prediction)

    prediction = re.sub(r'^\W+|\W+$', '', prediction)
    return prediction

