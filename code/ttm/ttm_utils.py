import json

total_opt = None
_questions = []

def all_questions():
    global _questions
    if _questions == []:
        with open('data/imdb-ecql.json', 'r') as f:
            ecql = json.load(f)
            _questions = [q['question'] for q in ecql]
    return _questions


_datasets = {
    'IMDB': ['imdb', None]
}


def dataset():
    fn, data = _datasets[total_opt.dataset]
    if not data:
        with open(f'data/{fn}-dataset.json', 'r') as f:
            data = json.load(f)
        _datasets[total_opt.dataset][1] = data
    return data
    

def gt_of(question_or_id):
    if isinstance(question_or_id, str):
        question = question_or_id
        for sample in dataset()['samples']:
            if sample['question'] == question:
                return [t for t in dataset()['template'] if t['template'] == sample['template']][0]
        print(f'Could not find gt for "{question}".')
    elif isinstance(question_or_id, int):
        qid = question_or_id
        for sample in dataset()['samples']:
            if sample['id'] == qid:
                return [t for t in dataset()['template'] if t['template'] == sample['template']][0]

    return None


def sample_of(question):
    for sample in dataset()['samples']:
        if sample['question'] == question:
            return sample
    print(f'Could not find sample for "{question}".')
    return None

MAX_CMP_RESULT_COUNT = 256
DATASET_FILEPATH = {
    'IMDB': 'data/imdb-dataset.json'
}
QL_TYPENAMES = {
    'AQL': 'aql',
    'ECQL': 'ecql',
    'SQL++': 'sqlpp'
}