"""
Main sequence to sequence class which conects
encoder-decoder model
"""
import torch.nn as nn

class Seq2Seq(nn.Module):
    """
    Seq2seq class
    """
    def __init__(self, encoder, decoder, name):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

    def forward(self, src_tokens, src_lengths, trg_tokens, teacher_forcing_ratio=0.5):
        """
        Run the forward pass for an encoder-decoder model.

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(src_len, batch)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            trg_tokens (LongTensor): tokens in the target language of shape
                `(tgt_len, batch)`, for teacher forcing
            teacher_forcing_ratio (float): teacher forcing probability

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - attention scores of shape `(batch, trg_len, src_len)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(trg_tokens, encoder_out,
                                   src_tokens=src_tokens,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_out