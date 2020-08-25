from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
from char_decoder import CharDecoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

import random

class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, no_char_decoder = False):
        super(NMT, self).__init__()

        # We keep separate model embeddings for source and target language
        self.model_embeddings_source = ModelEmbeddings(embed_size, vocab.src)
        self.model_embeddings_target = ModelEmbeddings(embed_size, vocab.tgt)

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # Initialize the encoder variable
        self.encoder = nn.LSTM(input_size = self.embed_size, self.hidden_size, bidirectional = True, bias = True)

        # input size is the embedding_size + attention output from the previous step(hidden_size)
        self.decoder = nn.LSTMCell(input_size = self.embed_size + self.hidden_size, hidden_size = self.hidden_size, bias =True)

        # Initialize the linear layer for the linear projections
        self.h_projection = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        # This is to compute the attention scores (Default is dot product)
        self.att_projection = nn.Linear(2 * self.hidden_size, self.hidden_size,  bias =False)

        # This is to compoute the output vector by combining attention output and decode hidden state
        self.combined_output_projection = nn.Linear(3 * self.hidden_size, self.hidden_size, bias = False)

        # Initiazlie the target vocab linear layer
        self.target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.tgt), bias = False)
        
        # Initialize dropout
        self.dropout = nn.Dropout(p = self.dropout_rate, inplace = False)

        # Whether to use char_decoder or not
        if not no_char_decoder:
           self.charDecoder = CharDecoder(hidden_size, target_vocab=vocab.tgt) 
        else:
           self.charDecoder = None


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        target_padded = self.vocab.tgt.to_input_tensor(target, device = self.device)
        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device = self.device)
        target_padded_chars = self.vocab.tgt.to_input_tensor_char(target, device = self.device)

        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hidden, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded_chars)

        # perform softamx
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim = -1)

        # Zero out probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum()

    
        if self.charDecoder is not None:
            max_word_len = target_padded_chars.shape[-1]
            # remove start of word character ?
            target_words = target_padded[1:].contiguous().view(-1)
            # view : (l, b, max_w_len) -> (l * b, max_w_len)
            target_chars = target_padded_chars[1:].contiguous().view(-1, max_word_len)
            target_outputs = combined_outputs.view(-1, 256)
    
            target_chars_oov = target_chars #torch.index_select(target_chars, dim=0, index=oovIndices)
            rnn_states_oov = target_outputs #torch.index_select(target_outputs, dim=0, index=oovIndices)
            oovs_losses = self.charDecoder.train_forward(target_chars_oov.t(), (rnn_states_oov.unsqueeze(0), rnn_states_oov.unsqueeze(0)))
            scores = scores - oovs_losses
    
        return scores

        
    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc_hiddens, dec_init_state = None, None

        X = self.model_embeddings_source(source_padded)
        X_packed = pack_padded_sequence(X, source_lengths)

        enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)
        (enc_hiddens, _) = pad_packed_sequence(enc_hiddens)
        enc_hiddens = enc_hiddens.permute(1,0,2)

        init_decoder_hidden = self.h_projection(torch.cat(lst_hidden[0], last_hidden[1]), dim = 1)
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim = 1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        # chop off the end token for max len sentences
        target_padded = target_pdaded[:-1]

        # initialize the decoder stae
        dec_state = dec_init_state
        
        # initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings_target(taret_padded)

        for Y_t in torch.split(Y, split_size_or_sections = 1):
            Y_t = Y_t.squeeze(0)
            Ybar_t = torch.cat([Y_t, o_prev], dim = -1)
            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)

    
    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.
        """"

        combined_output = None
        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state
        # batch matrix multiplication
        e_t = torch.bmm(enc_hiddens_proj, dec_hiddn.unsqueeze(2)).squeeze(2)

        # set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = 

