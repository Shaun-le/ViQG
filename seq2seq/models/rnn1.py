"""
Seq2seq based: Neural Machine Translation by Jointly Learning to Align and Translate
https://arxiv.org/abs/1409.0473
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq.models.conf import PAD_TOKEN, EOS_TOKEN, SOS_TOKEN
from seq2seq.models.layers import RNN, Embedding, Linear, LSTM, GRU

class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, vocabulary, device, embed_dim=512, hidden_size=512,
                 num_layers=2, dropout=0.5, bidirectional=True, cell_name='gru'):
        super().__init__()
        input_dim = len(vocabulary)
        self.vocabulary = vocabulary
        self.pad_id = vocabulary.stoi[PAD_TOKEN]
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = dropout
        self.bidirectional = bidirectional
        self.cell_name = cell_name
        self.device = device

        self.embed_tokens = Embedding(input_dim, self.embed_dim, self.pad_id)

        self.rnn_cell = RNN(cell_name)
        self.rnn = self.rnn_cell(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.,
            bidirectional=self.bidirectional
        )
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src_tokens, **kwargs):
        """
        Forward Encoder
        Args:
            src_tokens (LongTensor): (batch, src_len)
            src_lengths (LongTensor): (batch)
        Returns:
            x (LongTensor): (src_len, batch, hidden_size * num_directions)
            hidden (LongTensor): (batch, enc_hid_dim)
        """
        src_lengths = kwargs.get('src_lengths', '')
        src_tokens = src_tokens.t()

        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)  # (src_len, batch, embed_dim)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.cpu())

        packed_outputs, hidden = self.rnn(packed_x)  # hidden: (n_layers * num_directions, batch, hidden_size)

        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # input hidden for decoder is the final encoder hidden state
        # since rnn is bidirectional get last forward and backward hidden state

        last_forward = hidden[-2, :, :]
        last_backward = hidden[-1, :, :]
        hidden = torch.cat((last_forward, last_backward), dim=1)
        hidden = torch.tanh(self.linear_out(hidden))  # (batch, enc_hid_dim)

        return x, hidden


class Attention(nn.Module):
    """Attention"""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.linear = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        """
        Forward Attention Layer
        Args:
            hidden (LongTensor): (batch, dec_hid_dim)
            encoder_outputs (LongTensor): (src_len, batch, enc_hid_dim * 2)
            mask (LongTensor): (batch, src_len)
        Returns:
            attention (LongTensor): (batch, src_len)
        """

        batch = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, dec_hid_dim)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, src_len, enc_hid_dim * 2)

        energy = torch.tanh(self.linear(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, src_len, dec_hid_dim)
        energy = energy.permute(0, 2, 1)  # (batch, dec_hid_dim, src_len)

        v = self.v.repeat(batch, 1).unsqueeze(1)  # (batch, 1, dec_hid_dim)

        attention = torch.bmm(v, energy).squeeze(1)

        attention = attention.masked_fill(mask == 0, float('-inf'))

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decoder"""

    def __init__(self, vocabulary, device, embed_dim=512, hidden_size=512,
                 num_layers=2, dropout=0.5, max_positions=500, cell_name='gru'):
        super().__init__()
        self.vocabulary = vocabulary
        self.pad_id = vocabulary.stoi[PAD_TOKEN]
        self.sos_idx = vocabulary.stoi[SOS_TOKEN]
        self.eos_idx = vocabulary.stoi[EOS_TOKEN]

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.need_attn = True
        self.output_dim = len(vocabulary)

        self.dropout = dropout
        self.max_positions = max_positions
        self.device = device
        self.cell_name = cell_name

        # suppose encoder and decoder have same hidden size
        self.attention = Attention(self.hidden_size, self.hidden_size)
        self.embed_tokens = Embedding(self.output_dim, self.embed_dim, self.pad_id)

        self.rnn_cell = RNN(cell_name)
        self.rnn = self.rnn_cell(
            input_size=(hidden_size * 2) + embed_dim,
            hidden_size=hidden_size,
        )

        self.linear_out = Linear(
            in_features=(hidden_size * 2) + hidden_size + embed_dim,
            out_features=self.output_dim
        )

    def _decoder_step(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)  # (1, batch)

        x = self.embed_tokens(input)  # (1, batch, emb_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        attn = self.attention(hidden, encoder_outputs, mask)  # (batch, src_len)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        attn = attn.unsqueeze(1)  # (batch, 1, src_len)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, src_len, 2 * enc_hid_dim)

        weighted = torch.bmm(attn, encoder_outputs)  # (batch, 1, 2 * enc_hid_dim)

        weighted = weighted.permute(1, 0, 2)  # (1, batch, 2 * enc_hid_dim)

        rnn_input = torch.cat((x, weighted), dim=2)  # (1, batch, 2 * enc_hid_dim + embed_dim)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: (1, batch, dec_hid_dim)
        # hidden: (1, batch, dec_hid_dim)

        x = x.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        x = torch.cat((output, weighted, x), dim=1)
        output = self.linear_out(x)  # (batch, output_dim)

        return output, hidden.squeeze(0), attn.squeeze(1)

    def forward(self, trg_tokens, encoder_out, **kwargs):
        """
        Forward Decoder
        Args:
            trg_tokens (LongTensor): (trg_len, batch)
            Tuple (encoder_out):
                encoder_out (LongTensor): (src_len, batch, 2 * hidden_size)
                hidden (LongTensor): (batch, enc_hid_dim)
            src_tokens (LongTensor): (src_len, batch)
        Returns:
            outputs (LongTensor): (max_len, batch, output_dim)
            attentions (LongTensor): (max_len, batch, src_len)
        """
        encoder_out, hidden = encoder_out
        src_tokens = kwargs.get('src_tokens', '')
        teacher_ratio = kwargs.get('teacher_forcing_ratio', '')
        src_tokens = src_tokens.t()
        batch = src_tokens.shape[1]

        if trg_tokens is None:
            teacher_ratio = 0.
            inference = True
            trg_tokens = torch.zeros((self.max_positions, batch)).long(). \
                fill_(self.sos_idx). \
                to(self.device)
        else:
            trg_tokens = trg_tokens.t()
            inference = False

        max_len = trg_tokens.shape[0]

        # initialize tensors to store the outputs and attentions
        outputs = torch.zeros(max_len, batch, self.output_dim).to(self.device)
        attentions = torch.zeros(max_len, batch, src_tokens.shape[0]).to(self.device)

        # prepare decoder input(<sos> token)
        input = trg_tokens[0, :]

        mask = (src_tokens != self.pad_id).permute(1, 0)  # (batch, src_len)

        for i in range(1, max_len):

            # forward through decoder using inout, encoder hidden, encoder outputs and mask
            # get predictions, hidden state and attentions
            output, hidden, attention = self._decoder_step(input, hidden, encoder_out, mask)

            # save predictions for position i
            outputs[i] = output

            # save attention for position i
            attentions[i] = attention

            # if teacher forcing
            #   use actual next token as input for next position
            # else
            #   use highest predicted token
            input = trg_tokens[i] if random.random() < teacher_ratio else output.argmax(1)

            # if inference is enabled and highest predicted token is <eos> then stop
            # and return everything till position i
            if inference and input.item() == self.eos_idx:
                return outputs[:i]  # , attentions[:i]

        return outputs  # , attentions