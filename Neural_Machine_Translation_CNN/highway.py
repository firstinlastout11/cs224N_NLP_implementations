#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
     Highway network for Character-Level Network
     Highway Networks have a skip-connection controlled by a dynamic gate
    """
    def __init__(self, embed_size):
        """
            This initializes necessary layers and inputs to create the Highway network for the Char-level word embeddings
            Input shape : (bch, M_w)

        """
        super(Highway, self).__init__()
        # Embedding dimension of word
        self.embed_size = embed_size

        # Initiazlie the projection linear layer 
        self.projection = nn.Linear(self.embed_size, self.embed_size, bias=True)

        # Initialize the gate projection linear layer
        self.gate_projection = nn.Linear(self.embed_size, self.embed_size, bias=True)

    def forward(self, x_conv_out):
        """
            This serves to initialize the forward pass of a Highway network 
            This maps x_conv_out from the convlution step to x_highway
            The input shape: (bch, M_w) from CNN
        """
        # x_conv_out input goes through the projection linear layer with bias
        x_proj = self.projection(x_conv_out)

        # Then, it goes through the relu layer
        x_proj = F.relu(x_proj)

        # x_conv_out input also goest through the gate layer with bias
        x_gate = self.gate_projection(x_conv_out)

        # Then, it goes through the sigmoid layer
        x_gate = torch.sigmoid(x_gate)

        # perform the element-wise multiplication 
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        return x_highway

