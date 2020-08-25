#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        # store the variables
        self.char_pad_token_idx = vocab.char2id['<pad>']
        # Embedding size of each word-embedding
        self.embed_size = embed_size
        self.drop_rate = 0.3
        # Embedding size of each character-embedding
        self.char_embed_size = 50
        self.max_word_len = 21


        # Initialize the embedding layer to get the look-up table
        # num_embeddings: size of the character-dictionary
        # embeeding_dim : size of each character embedding vector
        self.char_embeddings = nn.Embedding(num_embeddings = len(vocab.char2id), embedding_dim = self.char_embed_size, padding_idx = self.char_pad_token_idx)

        # Initialize the CNN
        # input shape : (bch, M_ch, w_len)
        self.cnn = CNN(
            char_embed_size=self.char_embed_size, # Input-channels
            num_filters=embed_size,               # Output-channels
            max_word_length=self.max_word_len,
        )        
        # Initialize the Highway
        self.highway = Highway(embed_size)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(p = self.drop_rate)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """

        # Embedding layer
        # Input shape (sentence_length, batch_size, max_word_length) 
        char_embeddings = self.char_embeddings(input)

        # Embedidng shape (sentence_length, batch_size, max_word_length, char_embed_size = 50) 
        
        # store the shapes of x
        sent_len, batch_size, m_len, _ = char_embeddings.shape

        # reshape the char embeddings so that the batch_size = batch_size * sentence_length
        char_embeddings = char_embeddings.view(-1, self.max_word_len, self.char_embed_size)

        # since CNN requires input shape : (bch, M_ch, w_len), it needs to be transposed. The first and second dimensions are switched
        char_embeddings = torch.transpose(char_embeddings, 1,2)

        # Apply CNN layer to the char_embeddings matrix to acquire x_conv_out matrix as in the diagram
        x_conv_out = self.cnn(char_embeddings)

        # Apply the Highway layer to x_conv_out matrix to acquire x_highway matrix
        x_highway = self.highway(x_conv_out)

        # Apply dropout
        x_word_embed = self.dropout(x_highway)

        # Since the output of the highway layer has the shape: (bch, M_w) from CNN
        # It needs to have the form (sent_len, batch_size, M_w)
        x_word_embed = x_word_embed.view(sent_len, batch_size, -1)
        return x_word_embed
