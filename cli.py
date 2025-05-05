import click
import math
import numpy as np
import pandas as pd
#from nltk import word_tokenize
from torchtext.data import BucketIterator
from main import set_SEED
from parser_data.load_data import load_json
from parser_data.prepare_data import HandleDataset, HandleDatasetFill, HandleDatasetMCQ, HandleDatasetAG
from pre_trained.evaluation import compute_score
from pre_trained.preprocess import preprocess_function, preprocess_function_without_answer
from seq2seq.metrics import ComputeScorer
from seq2seq.models.conf import PAD_TOKEN
from seq2seq.models.seq2seq import Seq2Seq
from seq2seq.prediction import Predictor
from seq2seq.trainer import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer
import torch
import torch.nn as nn
from datasets import load_dataset
import torch.optim as optim
from IPython.display import display
import nltk
from underthesea import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')


@click.group()
def cli():
    pass

@cli.command('evaluateTNM')
@click.option('--model_name', type=click.Choice(('rnn','cnn','transformer')), default=None,
              help="Choice model")
#@click.option('--dataset', type=click.Choice(('ViNewsQA','ViQuAD','ViCoQA','ViMMRC1.0','ViMMRC2.0')),
#               default=None, help="the dataset used for training model")
@click.option('--dataset', default=None, help="the dataset used for training model")
@click.option('--attention', default='luong', type=click.Choice(('bahdanau','luong')), help='attention layer for rnn model')
@click.option('--batch_size', default=8, type=int, help='batch size')
@click.option('--epochs_num', default=20, type=int, help='number of epochs')
@click.option('--lr', default=0.001, type=float, help='learning rate')
@click.option('--cell_name', type=click.Choice(('lstm','gru')), default='gru')
@click.option('--task', type=click.Choice(('qg-aware','qg-agnostic','qag','mcq','fill')), default='qg-aware')
def _evaluateTNM(model_name, dataset, attention, batch_size, epochs_num, cell_name, task,lr):
    """
    Training and evaluate model for QG task in Vietnamese Text
    """
    print("data: ", dataset)
    print("model: ", model_name)
    print('--------------------------------')
    #train = load_json(f'datasets/{dataset}/train.json', dataset)
    #val = load_json(f'datasets/{dataset}/dev.json', dataset)
    #test = load_json(f'datasets/{dataset}/test.json', dataset)
    train = load_dataset(dataset,use_auth_token=True)['train']
    val = load_dataset(dataset,use_auth_token=True)['validation']
    test = load_dataset(dataset,use_auth_token=True)['test']
    def remove_nan_samples(dataset):
        return dataset.filter(lambda x: x['distract'] != '' and x['distract'] != 'nan')
    if task == 'qg-aware':
        dataset = HandleDataset(train, val, test)
    elif task == 'qg-agnostic':
        dataset = HandleDatasetAG(train, val, test)
    elif task == 'mcq':
        train = remove_nan_samples(train)
        val = remove_nan_samples(val)
        test = remove_nan_samples(test)
        dataset = HandleDatasetMCQ(train, val, test)
    elif task == 'fill':
        train = remove_nan_samples(train)
        val = remove_nan_samples(val)
        test = remove_nan_samples(test)
        dataset = HandleDatasetFill(train, val, test)
    dataset.load_data_and_fields()
    src_vocab, trg_vocab = dataset.get_vocabs()
    train_data, valid_data, test_data = dataset.get_data()
    print('--------------------------------')
    print(f"Training data: {len(train_data.examples)}")
    print(f"Evaluation data: {len(valid_data.examples)}")
    print(f"Testing data: {len(test_data.examples)}")
    print('--------------------------------')
    print(f'Question example: {train_data.examples[0].src}\n')
    print(f'Answer example: {train_data.examples[0].trg}')
    print('--------------------------------')
    print(f"Unique tokens in questions vocabulary: {len(src_vocab)}")
    print(f"Unique tokens in answers vocabulary: {len(trg_vocab)}")
    print('--------------------------------')

    set_SEED()

    if model_name == 'rnn' and attention == 'bahdanau':
        from seq2seq.models.rnn1 import Encoder, Decoder
    elif model_name == 'rnn' and attention == 'luong':
        from seq2seq.models.rnn2 import Encoder, Decoder
    elif model_name == 'cnn':
        from seq2seq.models.cnn import Encoder, Decoder
    elif model_name == 'transformer':
        from seq2seq.models.transformer import Encoder, Decoder, NoamOpt

    set_SEED()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'rnn' and attention == 'luong':
        encoder = Encoder(src_vocab, DEVICE, cell_name)
        decoder = Decoder(trg_vocab, DEVICE, cell_name)
    else:
        encoder = Encoder(src_vocab, DEVICE)
        decoder = Decoder(trg_vocab, DEVICE)

    model = Seq2Seq(encoder, decoder, model_name).to(DEVICE)

    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('--------------------------------')
    print(f'Model: {model_name}')
    print(f'Model input: context+answer')
    if model_name == 'rnn':
        print(f'Attention: {attention}')
        print('Cell name: ', cell_name)
    print(f'The model has {parameters_num:,} trainable parameters')
    print('--------------------------------')
    # create optimizer
    if model_name == 'transformer':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)


    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])
    trainer = Trainer(optimizer, criterion, batch_size, DEVICE)
    train_loss, val_loss = trainer.train(model, train_data, valid_data, 'datasets', num_of_epochs=epochs_num)

    val_ref = [list(filter(None, np.delete([sample["paragraph"], sample["answer"], sample["question"]], [0, 1]))) for
               sample in val]

    test_ref = [list(filter(None, np.delete([sample["paragraph"], sample["answer"], sample["question"]], [0, 1]))) for
                sample in test]

    val_trg = []
    test_trg = []
    trg_ = [val_trg, test_trg]
    for t in trg_:
        for i in test_ref:
            tmp = []
            for j in i:
                s = word_tokenize(str(j))
                tmp.append(s)
            t.append(tmp)

    val_src = [i.src for i in valid_data.examples]
    new_valid = [[val_src[i], [word_tokenize(val[i]["question"])]] for i in range(len(val))]
    test_src = [i.src for i in test_data.examples]
    new_test = [[test_src[i], [word_tokenize(test[i]["question"])]] for i in range(len(test))]

    valid_iterator, test_iterator = BucketIterator.splits(
        (valid_data, test_data),
        batch_size=8,
        sort_within_batch=True if model_name == 'rnn' else False,
        sort_key=lambda x: len(x.src),
        device=DEVICE)

    # evaluate model
    valid_loss = trainer.evaluator.evaluate(model, valid_iterator)
    test_loss = trainer.evaluator.evaluate(model, test_iterator)

    # calculate blue score for valid and test data
    predictor = Predictor(model, src_vocab, trg_vocab, DEVICE)

    valid_scorer = ComputeScorer()
    test_scorer = ComputeScorer()

    valid_scorer.data_score(new_valid, predictor)
    test_scorer.data_score(new_test, predictor)

    print(f'| Val. Loss: {valid_loss:.3f} | Test PPL: {math.exp(valid_loss):7.3f} |')
    print(f'| Val. Data Average BLEU1,BLEU2, BLEU3, BLEU4 score {valid_scorer.average_score()} |')
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print(f'| Test Data Average BLEU1,BLEU2, BLEU3, BLEU4 score {test_scorer.average_score()} |')

    r = {'ppl': [round(math.exp(test_loss), 3)],
         'BLEU-1': [test_scorer.average_score()[0] * 100],
         'BLEU-2': [test_scorer.average_score()[1] * 100],
         'BLEU-3': [test_scorer.average_score()[2] * 100],
         'BLEU-4': [test_scorer.average_score()[3] * 100],
         'ROUGE-1': [test_scorer.average_rouge_score_n()[0]],
         'ROUGE-2': [test_scorer.average_rouge_score_n()[1]],
         'ROUGE-L': [test_scorer.average_rouge_score() * 100]}

    df_result = pd.DataFrame(data=r)
    df_result.to_csv('results_traditional.csv')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df_result)

@cli.command('evaluate')
@click.option('--model', type=click.Choice(('ViT5','BARTPho')), default=None,
              help="Choice model")
@click.option('--dataset', type=click.Choice(('ViNewsQA','ViQuAD','ViCoQA','ViMMRC1.0','ViMMRC2.0')),
                default=None, help="the dataset used for training model")
@click.option('--answer', type=click.Choice(('y','n')), default='y', help="include an answer or not? 'y' for yes, 'n' for no.")
@click.option('--lr', default=1e-5, type=float, help='learning rate')
@click.option('--batch_size', default=4, type=int, help='batch size')
@click.option('--epochs_num', default=10, type=int, help='number of epochs')
@click.option('--path', type=str, default=None, help="The path to the location where you want to save the model, ignore if you don't want to save the model.")
def _evaluate(model,dataset,answer,lr, batch_size,epochs_num,path):
    print("data: ", dataset)
    print("model: ", model)
    print('--------------------------------')
    if model == 'ViT5':
        tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
        model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable-base')
        model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable-base")

    train = load_json(f'datasets/{dataset}/train.json', dataset)
    val = load_json(f'datasets/{dataset}/dev.json', dataset)
    test = load_json(f'datasets/{dataset}/test.json', dataset)

    if answer == 'y':
        tokenized_train = train.map(function=preprocess_function, batched=True,remove_columns=['contexts', 'answers', 'questions'],fn_kwargs={"tokenizer": tokenizer}, num_proc=8)
        tokenized_dev = val.map(function=preprocess_function, batched=True, remove_columns=['contexts', 'answers', 'questions'],fn_kwargs={"tokenizer": tokenizer},num_proc=8)
    else:
        tokenized_train = train.map(function=preprocess_function_without_answer, batched=True,remove_columns=['contexts', 'answers', 'questions'],fn_kwargs={"tokenizer": tokenizer}, num_proc=8)
        tokenized_dev = val.map(function=preprocess_function_without_answer, batched=True,remove_columns=['contexts', 'answers', 'questions'],fn_kwargs={"tokenizer": tokenizer}, num_proc=8)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments("tmp/",
                                             do_train=True,
                                             do_eval=True,
                                             num_train_epochs=epochs_num,
                                             learning_rate=lr,
                                             warmup_ratio=0.05,
                                             weight_decay=0.01,
                                             per_device_train_batch_size=batch_size,
                                             per_device_eval_batch_size=16,
                                             predict_with_generate=True,
                                             group_by_length=True,
                                             save_total_limit=1,
                                             gradient_accumulation_steps=16,
                                             eval_steps=50,
                                             evaluation_strategy="steps",
                                             )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        eval_dataset=tokenized_dev
    )

    trainer.train()

    if path != "":
        trainer.save_model(path)

    rouge, bleu_1, bleu_2, bleu_3, bleu_4 = compute_score(answer,test,model,tokenizer)

    r = {'BLEU-1': bleu_1 * 100,
         'BLEU-2': bleu_2 * 100,
         'BLEU-3': bleu_3 * 100,
         'BLEU-4': bleu_4 * 100,
         'ROUGE-1': rouge[0]['rouge1'] * 100,
         'ROUGE-2': rouge[1]['rouge2'] * 100,
         'ROUGE-L': rouge[2]['rougeL'] * 100}

    df_result = pd.DataFrame(data=r, index=[0])
    #df_result.to_csv('results_pretrained.csv')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df_result)


if __name__ == '__main__':
    cli()
