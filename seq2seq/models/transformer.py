"""
Seq2seq based: Attention Is All You Need
https://arxiv.org/abs/1706.03762
"""
import math
import torch
import torch.nn as nn
import torch.onnx.operators
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, vocabulary, device, embed_dim=512, layers=2,
                 heads=8, pf_dim=2048, dropout=0.5, max_positions=5000):
        super().__init__()
        input_dim = len(vocabulary)
        self.padding_idx = vocabulary.stoi['<pad>']
        self.dropout = dropout
        self.device = device

        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(input_dim, embed_dim)
        self.embed_positions = PositionalEmbedding(embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList([EncoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

    def forward(self, src_tokens, **kwargs):
        """
        Forward pass for transformer encoder
        Args:
            src_tokens (LongTensor): (batch, src_len)
        Returns:
            x (LongTensor): (batch, src_len, embed_dim)
        """
        src_mask = (src_tokens != self.padding_idx).unsqueeze(1).unsqueeze(2)

        x = self.embed_tokens(src_tokens) * self.scale
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class EncoderLayer(nn.Module):
    """EncoderLayer"""
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, src_mask):
        """
        Forward pass for transformer encoder layer
        Args:
            src_tokens (LongTensor): (batch, src_len, embed_dim)
            src_mask (LongTensor): (batch, src_len)
        Returns:
            x (LongTensor): (batch, src_len, embed_dim)
        """
        x = self.layer_norm(src_tokens + self.dropout(self.self_attn(src_tokens, src_tokens, src_tokens, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, vocabulary, device, embed_dim=512, layers=2,
                 heads=8, pf_dim=2048, dropout=0.5, max_positions=5000):
        super().__init__()

        output_dim = len(vocabulary)
        self.pad_id = vocabulary.stoi['<pad>']
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.max_positions = max_positions

        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(output_dim, embed_dim)
        self.embed_positions = PositionalEmbedding(embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList([DecoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

        self.linear_out = nn.Linear(embed_dim, output_dim)

    def make_masks(self, src_tokens, trg_tokens):
        src_mask = (src_tokens != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg_tokens != self.pad_id).unsqueeze(1).unsqueeze(3).byte()
        trg_len = trg_tokens.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).byte()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, trg_tokens, encoder_out, **kwargs):
        """
        Forward pass for transformer decoder
        Args:
            trg_tokens (LongTensor): (batch, trg_len)
            encoder_out (LongTensor): (batch, src_len, embed_dim)
            src_tokens (LongTensor): (batch, src_len)
        Returns:
            x (LongTensor): (batch, trg_len, output_dim)
        """
        src_tokens = kwargs.get('src_tokens', '')
        src_mask, trg_mask = self.make_masks(src_tokens, trg_tokens)

        #print(trg_tokens.shape) #batch_size = 12
        x = self.embed_tokens(trg_tokens) * self.scale #[12, 296, 512]

        x += self.embed_positions(trg_tokens)#[1, 100, 512]
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, encoder_out, trg_mask, src_mask)

        return self.linear_out(x)

class DecoderLayer(nn.Module):
    """DecoderLayer"""
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.src_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embed_trg, embed_src, trg_mask, src_mask):
        """
        Forward pass for transformer decoder layer
        Args:
            embed_trg (LongTensor): (batch, trg_len, embed_dim)
            embed_src (LongTensor): (batch, src_len, embed_dim)
            trg_mask (LongTensor): (batch, trg_len)
            src_mask (LongTensor): (batch, src_len)
        Returns:
            x (LongTensor): (batch, trg_len, embed_dim)
        """
        x = self.layer_norm(embed_trg + self.dropout(self.self_attn(embed_trg, embed_trg, embed_trg, trg_mask)))
        x = self.layer_norm(x + self.dropout(self.src_attn(x, embed_src, embed_src, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x

class MultiHeadedAttention(nn.Module):
    """MultiHeadedAttention"""
    def __init__(self, embed_dim, heads, dropout, device):
        super().__init__()
        assert embed_dim % heads == 0
        self.attn_dim = embed_dim // heads
        self.heads = heads
        self.dropout = dropout

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.attn_dim])).to(device)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for transformer decoder layer
        Args:
            query (LongTensor): (batch, sent_len, embed_dim)
            key (LongTensor): (batch, sent_len, embed_dim)
            value (LongTensor): (batch, sent_len, embed_dim)
            mask (LongTensor): (batch, sent_len)
        Returns:
            x (LongTensor): (batch, sent_len, embed_dim)
        """
        batch_size = query.shape[0]

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        Q = Q.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)
        K = K.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)
        V = V.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # (batch, heads, sent_len, sent_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1) # (batch, heads, sent_len, sent_len)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        x = torch.matmul(attention, V) # (batch, heads, sent_len, attn_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # (batch, sent_len, heads, attn_dim)
        x = x.view(batch_size, -1, self.heads * (self.attn_dim)) # (batch, sent_len, embed_dim)
        x = self.linear_out(x)

        return x

class PositionwiseFeedforward(nn.Module):
    """PositionwiseFeedforward"""
    def __init__(self, embed_dim, pf_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        """
        PositionwiseFeedforward
        Args:
            x (LongTensor): (batch, src_len, embed_dim)
        Returns:
            x (LongTensor): (batch, src_len, embed_dim)
        """
        x = torch.relu(self.linear_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.linear_2(x)

class PositionalEmbedding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=500):
        super().__init__()
        pos_embed = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):

        return Variable(self.pos_embed[:, :x.size(1)], requires_grad=False)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=512, factor=1, warmup=2000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.param_groups = optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
