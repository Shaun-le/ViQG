import argparse
import random
import numpy as np
import torch
def parse_args():
    """Add arguments to parser"""
    parser = argparse.ArgumentParser(description='Verbalization dataset baseline models.')
    parser.add_argument('--model', default='rnn', type=str,
                        choices=['rnn', 'cnn', 'transformer'], help='model to train the dataset')
    parser.add_argument('--input', default='CA', type=str,
                        choices=['CA'], help='use question as input')
    parser.add_argument('--attention', default='luong', type=str,
                        choices=['bahdanau', 'luong'], help='attention layer for rnn model')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs_num', default=30, type=int, help='number of epochs')
    args = parser.parse_args()
    return args

def set_SEED():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Checkpoint(object):
    """Checkpoint class"""
    @staticmethod
    def save(model,cell, path):
        """Save model using name"""
        name_tmp = model.name+"_"+ cell if model.name=='rnn' else model.name
        name = f'{name_tmp}.pt'
        torch.save(model.state_dict(), path+name)

    @staticmethod
    def load(model,path, name):
        """Load model using name"""
        #name = f'{model.name}.pt'
        model.load_state_dict(torch.load(path+name))
        return model