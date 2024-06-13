import tiktoken
import time
from openai import OpenAI

with open('../keys/openai_key', 'r') as f:
    api_key = f.readline().strip()

with open('../keys/openai_org_id', 'r') as f:
    organization = f.readline().strip()

client = OpenAI(api_key=api_key, organization=organization)

tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-1106')
#tokenizer = tiktoken.encoding_for_model('gpt-4-1106-preview')

def call(message, max_tokens=100):
    messages = [{'role': 'user', 'content': message}]

    while True:
        try:
            response = client.chat.completions.create(
                            model='gpt-3.5-turbo-1106',
#                            model='gpt-4-1106-preview',
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=0.2
                            )
            break
        except Exception as e:
            time.sleep(2)
            print('Errrrrrrrrrrrrrrrrrr', str(e))
            print(len(tokenizer.encode(messages[0]['content'])))

    prediction = response.choices[0].message.content

    return prediction

