"""
Seq2seq based: Convolutional Sequence to Sequence Learning
https://arxiv.org/abs/1705.03122
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq.models.conf import PAD_TOKEN
from seq2seq.models.layers import Embedding, Linear, Conv1d, PositionalEmbedding, LearnedPositionalEmbedding

def extend_conv_spec(convolutions):
    """
    Extends convolutional spec with default residual if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)

class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, vocabulary, device, embed_dim=512, convolutions=((512, 3),) * 3,
                 dropout=0.5, max_positions=5000):
        super().__init__()
        self.vocabulary = vocabulary
        input_dim = len(vocabulary)
        self.padding_idx = vocabulary.stoi[PAD_TOKEN]
        self.dropout = dropout
        self.device = device

        self.embed_tokens = Embedding(input_dim, embed_dim, self.padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, self.padding_idx)

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.embed2inchannels = Linear(embed_dim, in_channels)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels * 2,
                          kernel_size=kernel_size,
                          padding=padding)
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.inchannels2embed = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, **kwargs):
        """
        Forward pass for convolutional encoder
        Args:
            src_tokens (LongTensor): (batch, src_len)
        Returns:
            conved (LongTensor): (batch, src_len, embed_dim)
            combined (LongTensor): (batch, src_len, embed_dim)
        """
        # embed tokens and positions
        embedded = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        embedded = F.dropout(embedded, p=self.dropout, training=self.training)

        conv_input = self.embed2inchannels(embedded) # (batch, src_len, in_channels)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(self.padding_idx) # (batch, src_len)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = conv_input.permute(0, 2, 1) # (batch, in_channels, src_len)
        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=1)

            # apply residual connection
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        conved = self.inchannels2embed(x.permute(0, 2, 1))

        if encoder_padding_mask is not None:
            conved = conved.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        combined = (conved + embedded) * math.sqrt(0.5)

        return conved, combined
class Attention(nn.Module):
    """Attention"""
    def __init__(self, conv_channels, embed_dim):
        super().__init__()
        self.linear_in = Linear(conv_channels, embed_dim)
        self.linear_out = Linear(embed_dim, conv_channels)

    def forward(self, conved, embedded, encoder_out, encoder_padding_mask):
        """
        Forward Attention Layer
        Args:
            conved (LongTensor): (batch, conv_channels, trg_len)
            embedded (LongTensor): (batch, trg_len, embed_dim)
            encoder_out (encoder_conved, encoder_combined): (batch, src_len, embed_dim)
            encoder_padding_mask(LongTensor): (batch, src_len)
        Returns:
            attended_combined (LongTensor): (batch, conv_channels, trg_len)
            attention (LongTensor): (batch, trg_len, src_len)
        """
        encoder_conved, encoder_combined = encoder_out

        conved_emb = self.linear_in(conved.permute(0, 2, 1)) # (batch, trg_len, embed_dim)
        combined = (conved_emb + embedded) * math.sqrt(0.5) # (batch, trg_len, embed_dim)

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1)) # (batch, trg_len, src_len)

        # don't attend over padding
        energy = energy.float().masked_fill(encoder_padding_mask.unsqueeze(1), float('-inf'))

        attention = F.softmax(energy, dim=2)

        attended_encoding = torch.matmul(attention, encoder_combined) # (batch, trg_len, embed_dim)
        attended_encoding = self.linear_out(attended_encoding) # (batch, trg_len, conv_channels)

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * math.sqrt(0.5)

        return attended_combined, attention
class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, vocabulary, device, embed_dim=512, convolutions=((512, 3),) * 3,
                 dropout=0.5, max_positions=5000):
        super().__init__()

        self.vocabulary = vocabulary
        self.dropout = dropout
        self.device = device
        self.max_positions = max_positions

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        output_dim = len(vocabulary)
        self.padding_idx = vocabulary.stoi[PAD_TOKEN]

        self.embed_tokens = Embedding(output_dim, embed_dim, self.padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, self.padding_idx)

        self.embed2inchannels = Linear(embed_dim, in_channels)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            self.convolutions.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels * 2,
                          kernel_size=kernel_size)
            )
            self.attention.append(Attention(out_channels, embed_dim))
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.inchannels2embed = Linear(in_channels, embed_dim)
        self.linear_out = Linear(embed_dim, output_dim)

    def forward(self, trg_tokens, encoder_out, **kwargs):
        """
        Forward pass for convolutional decoder
        Args:
            trg_tokens (LongTensor): (batch, trg_len)
            encoder_out (encoder_conved, encoder_combined): (batch, src_len, embed_dim)
            src_tokens (LongTensor): (batch, src_len)
        Returns:
            outputs (LongTensor): (batch, trg_len, output_dim)
            avg_attn_scores (LongTensor): (batch, trg_len, src_len)
        """
        src_tokens = kwargs.get('src_tokens', '')
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        # embed tokens and positions
        embedded = self.embed_tokens(trg_tokens) + self.embed_positions(trg_tokens)
        embedded = F.dropout(embedded, p=self.dropout, training=self.training) # (batch, trg_len, embed_dim)

        conv_input = self.embed2inchannels(embedded) # (batch, trg_len, in_channels)

        x = conv_input.permute(0, 2, 1) # (batch, in_channels, trg_len)

        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(self.projections, self.convolutions, self.attention,
                                                    self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            # add padding
            padding = torch.zeros(x.shape[0],
                                  x.shape[1],
                                  conv.kernel_size[0] - 1).fill_(self.padding_idx).to(self.device)
            x = torch.cat((padding, x), dim=2)
            x = conv(x)
            x = F.glu(x, dim=1)

            # attention
            x, attn_scores = attention(x, embedded, encoder_out, encoder_padding_mask)

            if not self.training:
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)

            # apply residual connection
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        conved = self.inchannels2embed(x.permute(0, 2, 1)) # (batch, trg_len, embed_dim)
        conved = F.dropout(conved, p=self.dropout, training=self.training)

        outputs = self.linear_out(conved)

        return outputs