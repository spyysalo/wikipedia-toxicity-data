import json
import tensorflow_datasets

keys = [
    'toxicity',
    'severe_toxicity',
    'identity_attack',
    'insult',
    'obscene',
    'threat',
]

d = tensorflow_datasets.load('wikipedia_toxicity_subtypes')

for s in { 'train', 'test' }:
    with open(f'{s}.tsv', 'w') as out:
        print('\t'.join(keys + ['text']), file=out)
        for x in d[s]:
            text = x['text'].numpy().decode('utf-8')
            values = [str(int(x[k].numpy())) for k in keys]
            print('\t'.join(values + [json.dumps(text)]), file=out)
