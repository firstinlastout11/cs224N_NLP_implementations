#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        
        # Initialize as an nn.Module
        super(CharDecoder, self).__init__()
        
        # Initiazlie the decoder LSTM: unidirectional, LSTM
        self.charDecoder = nn.LSTM(input_size = char_embedding_size, hidden_size = hidden_size)

        # Initialize the projection linear layer
        self.char_output_projection = nn.Linear(in_features = hidden_size,
                                                out_features = len(target_vocab.char2id),
                                                bias = True)

        # Initialize the embedding matrix of character embeddings
        self.decoderCharEmb = nn.Embedding(num_embeddings = len(target_vocab.char2id),
                                            embedding_dim = char_embedding_size,
                                            padding_idx = target_vocab.char2id['<pad>'])
        
        # Initialize the vocabulary for the target language
        self.vocab = target_vocab.char2id
        self.vocab_reverse = target_vocab.id2char
        self.target_vocab = target_vocab

    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### TODO - Implement the forward pass of the character decoder.

        # Get the embedding matrix of the given input
        char_embedding = self.decoderCharEmb(input)

        # Apply the LSTM to the input
        dec_state = self.charDecoder(char_embedding, dec_hidden)

        # Split the hidden states and cell states
        (dec_hiddens, dec_hidden) = dec_state

        # Apply the output projection to get the scores
        scores = self.char_output_projection(dec_hiddens)

        # Return the scores and dec_state (afte rthe LSTM)
        return (scores, dec_hidden)



    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).


        # We pass the input sequence x_1,..,x_n (along with the initial states h_0 and c_0 from the combined output vector)
        # into the CharDecoderLSTM, thus obtaining scores s_1,...,s_n

        # the input sequence x_1,..,x_n
        input_padded = char_sequence[:-1]

        # Apply the forward step to inputs and dec_hidden to acquire scores and dec_state
        scores, _ = self.forward(input_padded, dec_hidden)

        # the target sequence x_2,...,x_n+1
        # This has shape: (length, batch) -> needs to be a list
        target_sequence = char_sequence[1:]

        # flatten the target matrix to feed into the cross entropy loss -> shape: (batch_size)
        target_sequence = torch.flatten(target_sequence)

        # recall that cross entropy loss combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        # scores has shape: (length, batch, self.vocab_size)
        # cross entropoy expects (batch, C) so we want to flatten the first two elements
        scores = scores.view(-1, scores.shape[-1])

        # Apply the cross entropy loss
        # Make sure that the padding characters do not contribute to the cross-entropy loss (ignore_index)
        # reducion: `sum`
        ce_loss = nn.CrossEntropyLoss(ignore_index = self.vocab['<pad>'],
                                    reduction = 'sum')

        loss = ce_loss(input = scores, target = target_sequence)

        return loss





    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.


        num_batches = initialStates[0].size(1)
        # Initialize the list to store the results
        decodedWords = [['', False] for _ in range(num_batches)]

        # Assign inital_States to the dec_hidden
        dec_hidden = initialStates

        # Initialize the initial input, that contains the start_of_word char for all batch
        # shape (len, batch)
        outputs = torch.tensor([[self.vocab['{'] for _ in range(num_batches)]], device = device)

        # Iterate through the max_length
        for _ in range(max_length):

            # Apply the forward step to the input and the initialStates
            scores, dec_hidden = self.forward(input= outputs, dec_hidden = dec_hidden)

            # perform softmax to the scores
            probs = F.softmax(scores, dim = -1)

            # pefrorm argmax to get the indice to the char with the highest probability
            outputs = torch.argmax(probs, dim = -1)
            
            # For each output, find the corresponding character and append it to the decodedWords
            for i in range(num_batches):
                if decodedWords[i][1] == False:
                    if self.vocab_reverse[int(outputs[0][i])] != '}':
                        decodedWords[i][0] += self.vocab_reverse[int(outputs[0][i])]
                    else:
                        decodedWords[i][1] = True

        decodedWords = [x[0] for x in decodedWords]

        return decodedWords

