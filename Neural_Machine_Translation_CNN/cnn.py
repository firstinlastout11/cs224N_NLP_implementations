#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
        This class is to implement the convolutional network that takes the input word embeddings.
        This performs a 1-dimensional convlution and Max pooling
    """

    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size = 5, padding =1):
        """
            Args:
            - embed_size (int): embedding size for characters
  
            - k (int): kernel size(window size): dictates the size of the window used to compute features

            - f : number of filters ( word_embed_size)
        """

        # input shape : (bch, char_embed, w_len)
        # output shape : (bch, num_filters, cov_lens)
        super(CNN, self).__init__()

        self.char_embed_size = char_embed_size # num input channels
        self.num_filters     = num_filters     # num output channels
        self.kernel_size     = kernel_size
        self.max_word_length = max_word_length
        self.padding = padding

        self.conv = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding = self.padding,
            bias=True
        )
        # Since we want to take the maximum across the second dimension, we can simply use AdaptiveMaxPool1d
        # Applies a 1D adaptive max pooling over an input signal composed of several input planes.
        # The output size is H, for any input size. The number of output features is equal to the number of input planes.
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input):
        """
            This function implements the forward step of the CNN layer
            Input shape : (bch, M_ch, w_len)
            Output shape : (bch, M_w)
        """

        # Apply Conv1d Layer to the input to get x_conv
        # x_conv shape : (bch, M_w, w_len-k+1)
        x_conv = self.conv(input)

        # Apply relu non-linearity
        x_conv = F.relu(x_conv)

        # Apply max_pool
        # x_conv shape: (bch, M_w)
        x_conv = self.maxpool(x_conv).squeeze(-1)



        return x_conv
        


