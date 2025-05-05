# Question Generation (QG): An Experimental Study for Vietnamese Text

## Directory
Please note that you should prepare a folder to store the data as shown below:

    ├── datasets/
      ├── ViNewsQA/
        ├── train.json
        ├── dev.jon
        ├── test.jon
      ├── ViQuAD/
    ├── parser_data/
    ├── seq2seq/
    ├── cli.py
    └── main.py

## Data
The available datasets for this source code include: [ViNewsQA](https://arxiv.org/abs/2006.11138), [ViQuAD](https://arxiv.org/abs/2009.14725), 
[ViCoQA](https://arxiv.org/abs/2105.01542), [ViMMRC1.0](https://arxiv.org/abs/2008.08810), and [ViMMRC2.0](https://arxiv.org/abs/2303.18162).

*If you want to train a model on your own dataset, convert that dataset to a format similar to one of the five datasets provided.

## Usage
### Install
```
git clone https://github.com/Shaun-le/ViQG.git
cd ViQG
```
### Prerequisite
To install dependencies, run:
```
pip install -r requirements.txt
```
### CLI
To proceed with model training, please run the following code snippets:
- ViT5 and BARTpho
```
python cli.py _evaluate --model_name 'ViT5' --dataset 'ViNewsQA' --answer 'y'
```
**Note**

--dataset: name of dataset

--answer: include an answer or not? 'y' for yes, 'n' for no. default='y'.

```
python cli.py _evaluate --model_name 'ViT5' --dataset 'ViNewsQA' --lr 1e-5 --batch_size 16 --epochs_num 10
```
## System

    Comming soon!
    
## Citation

```
@inproceedings{inproceedings,
author = {Quoc-Hung, Pham and Le, Huu-Loi and Minh, Dang and Tran, Khang and Vu, Huy-The and Nguyen, Minh-Tien and Phan, Xuan-Hieu},
year = {2023},
month = {12},
pages = {324-329},
title = {Question Generation: An Experimental Study for Vietnamese Text},
doi = {10.1109/RIVF60135.2023.10471875}
}
```
