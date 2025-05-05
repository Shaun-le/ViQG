import nltk
import numpy as np
from datasets import load_metric
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import tqdm
import spacy
nlpvi = spacy.load('vi_core_news_lg')
smoothie = SmoothingFunction().method4

class ComputeScorer(object):
    """Blue scorer class"""

    def __init__(self):
        self.results = []
        self.predictions = []
        self.references = []
        self.score = 0
        self.bleu_2 = 0
        self.bleu_3 = 0
        self.bleu_4 = 0
        self.rouge_1 = 0
        self.rouge_2 = 0
        self.rouge_score = 0

        self.instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        bleu_1 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0),
                                                         smoothing_function=SmoothingFunction().method4)
        bleu_2 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0),
                                                         smoothing_function=SmoothingFunction().method4)
        bleu_3 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0),
                                                         smoothing_function=SmoothingFunction().method4)
        bleu_4 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                                                         smoothing_function=SmoothingFunction().method4)
        return bleu_1, bleu_2, bleu_3, bleu_4

    def example_score_rouge(self, reference, hypothesis):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = []
        for i in reference:
            scores.append(scorer.score(i,hypothesis)['rougeL'][-1])
        return np.max(scores)

    def data_score(self, data, predictor):
        """Score complete list of data"""
        results_prelim = []
        for example in tqdm.tqdm(data):
            src = example[0]

            reference = [[string.lower() for string in sublist] for sublist in example[1]]
            # and calculate bleu score average of all hypothesis = predictor.predict(example.src)
            hypothesis = predictor.predict(src)
            bleu_1, bleu_2, bleu_3, bleu_4 = self.example_score(reference, hypothesis)
            rouge_score = self.example_score_rouge([' '.join(i) for i in reference], ' '.join(hypothesis))
            self.references.append([' '.join(i) for i in reference])
            self.predictions.append(' '.join(hypothesis))
            f = open("result.txt", "a", encoding = 'utf-8')
            f.write('Question: ' + " ".join(src) + '\n')
            for i in range(len(reference)):
                f.write('Reference_{}: '.format(i) + " ".join(reference[i]) + '\n')
            f.write('Hypothesis: ' + " ".join(hypothesis) + '\n')
            f.write('BLEU-1: ' + str(bleu_1 * 100) + '\n')
            f.write('BLEU-2: ' + str(bleu_2 * 100) + '\n')
            f.write('BLEU-3: ' + str(bleu_3 * 100) + '\n')
            f.write('BLEU-4: ' + str(bleu_4 * 100) + '\n')
            f.write('ROUGE-L: ' + str(rouge_score * 100) + '\n\n')
            f.close()

            results_prelim.append({
                'c_a': '"' + str(src) + '"',
                'reference': reference,
                'hypothesis': hypothesis,
                'bleu_1': bleu_1,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3,
                'bleu_4': bleu_4,
                'rouge_score': rouge_score,

            })

        results = [max((v for v in results_prelim if v['c_a'] == x), key=lambda y: y['bleu_1']) for x in
                   set(v['c_a'] for v in results_prelim)]

        with open('result_output.txt', 'w', encoding = 'utf-8') as f:
            for elem in results:
                f.write("%s\n" % elem)
                self.results.append(elem)
                self.score += elem['bleu_1']
                self.bleu_2 += elem['bleu_2']
                self.bleu_3 += elem['bleu_3']
                self.bleu_4 += elem['bleu_4']
                self.rouge_score += elem['rouge_score']
                self.instances += 1
        return (self.score / self.instances, self.bleu_2 / self.instances,
                self.bleu_3 / self.instances, self.bleu_4 / self.instances,
                self.rouge_score / self.instances)

    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances, self.bleu_2 / self.instances, self.bleu_3 / self.instances, self.bleu_4 / self.instances

    def average_rouge_score(self):
        """Return rouge average score"""
        return self.rouge_score / self.instances

    def average_rouge_score_n(self):
        metrics = load_metric('rouge')
        metrics.add_batch(predictions=self.predictions, references=self.references)
        metrics.compute()
        results = [{k: v.mid.fmeasure} for k,v in metrics.compute(predictions=self.predictions, references=self.references).items()]
        self.rouge_1 = results[0]['rouge1']*100
        self.rouge_2 = results[1]['rouge2'] * 100
        return self.rouge_1, self.rouge_2

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.score = 0
        self.instances = 0
        self.predictions = []
        self.references = []
        self.rouge_1 = 0
        self.rouge_2 = 0