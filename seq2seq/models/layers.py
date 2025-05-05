import torch.nn as nn
import torch


def RNN(cell_name):
    if cell_name.lower() == 'lstm':
        return LSTM
    elif cell_name.lower() == 'gru':
        return GRU
    else:
        raise ValueError(f"Unsupported RNN Cell: {cell_name}")


def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    """Linear layer"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    """LSTM layer"""
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def GRU(input_size, hidden_size, **kwargs):
    """GRU layer"""
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Conv1d(in_channels, out_channels, kernel_size, padding=0):
    """Conv1d"""
    m = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.bias, 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    """PositionalEmbedding"""
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class LearnedPositionalEmbedding(nn.Embedding):
    """LearnedPositionalEmbedding"""

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        """Input size [bsz x seqlen]"""
        # Replace non-padding symbols with their position numbers.
        # Position numbers begin at padding_idx+1. Padding symbols are ignored.
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return super().forward(positions)