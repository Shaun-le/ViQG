import json
import tqdm
from datasets import Dataset
def load_json(data_path:str, dataset:str):
    questions = []
    contexts, answers = [], []
    if dataset in ['ViNewsQA','ViQuAD']:
        with open(data_path) as f:
            data = json.load(f)
            del data["version"]
        for i in tqdm.tqdm(range(len(data['data']))):
            for paragraph in data['data'][i]['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    context = paragraph['context']
                    answer = qa['answers'][0]['text']
                    answers.append(answer)
                    contexts.append(context)
                    questions.append(question)
    elif dataset in ['ViMMRC1.0','ViMMRC2.0']:
        with open(data_path) as f:
            data_list = json.load(f)
        for data in tqdm.tqdm(data_list):
            article = data['article']
            questions_list = data['questions']
            options_list = data['options']
            answers_list = data['answers']

            for i, question in enumerate(questions_list):
                answer = options_list[i][ord(answers_list[i]) - 65]
                questions.append(question)
                contexts.append(article)
                answers.append(answer)
    else:
        with open(data_path) as f:
            data = json.load(f)
            del data["version"]
        for i in tqdm.tqdm(range(len(data['data']))):
            for question in data['data'][i]['questions']:
                questions.append(question['input_text'])
                contexts.append(data['data'][i]['story'])
            for answer in data['data'][i]['answers']:
                answers.append(answer['input_text'])
    dict_obj = {'contexts':contexts, 'answers':answers, 'questions':questions}
    datasets = Dataset.from_dict(dict_obj)
    return datasets