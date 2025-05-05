from datasets import load_metric
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm_notebook
import torch
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import nltk
from nltk import word_tokenize
from pre_trained.preprocess import preprocess_function, preprocess_function_without_answer
smoothie = SmoothingFunction().method4

def example_score(reference, hypothesis):
  bleu_1 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0),
                                                         smoothing_function=SmoothingFunction().method4)
  bleu_2 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0),
                                                         smoothing_function=SmoothingFunction().method4)
  bleu_3 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0),
                                                         smoothing_function=SmoothingFunction().method4)
  bleu_4 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                                                         smoothing_function=SmoothingFunction().method4)
  return bleu_1, bleu_2, bleu_3, bleu_4
def compute_score(answer,testset,model,tokenizer):
    if answer == "1":
        tokenized_test = testset.map(function=preprocess_function, batched=True,remove_columns=['contexts', 'answers', 'questions'],fn_kwargs={"tokenizer": tokenizer}, num_proc=8)
    else:
        tokenized_test = testset.map(function=preprocess_function_without_answer, batched=True,remove_columns=['contexts', 'answers', 'questions'],fn_kwargs={"tokenizer": tokenizer}, num_proc=8)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    metrics = load_metric('rouge')
    max_target_length = 256
    dataloader = torch.utils.data.DataLoader(tokenized_test, collate_fn=data_collator, batch_size=16)

    predictions = []
    references = []
    for i, batch in enumerate(tqdm_notebook(dataloader)):
        outputs = model.to('cuda').generate(
            input_ids=batch['input_ids'].to('cuda'),
            max_length=max_target_length,
            attention_mask=batch['attention_mask'].to('cuda'),
        )
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in
                       outputs]
            labels = np.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)
            actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in
                       labels]
        predictions.extend(outputs)
        references.extend(actuals)
        metrics.add_batch(predictions=outputs, references=actuals)

    metrics.compute()

    rouge = [{k: v.mid.fmeasure} for k, v in metrics.compute(predictions=predictions, references=references).items()]

    bleu_1, bleu_2, bleu_3, bleu_4 = [],[],[],[]
    for i in range(len(references)):
        b1, b2, b3, b4 = example_score([word_tokenize(references[i])], word_tokenize(predictions[i]))
        bleu_1.append(b1)
        bleu_2.append(b2)
        bleu_3.append(b3)
        bleu_4.append(b4)

    return rouge, np.mean(bleu_1), np.mean(bleu_2), np.mean(bleu_3), np.mean(bleu_4)