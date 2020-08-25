# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# CS224N 2018-19: Homework 5
# nmt.py: NMT Model
# Pencheng Yin <pcyin@cs.cmu.edu>
# Sahil Chopra <schopra8@stanford.edu>
# """

# import math
# from typing import List

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def pad_sents_char(sents, char_pad_token):
#     """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
#     @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
#         from `vocab.py`
#     @param char_pad_token (int): index of the character-padding token
#     @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
#         than the max length sentence/word are padded out with the appropriate pad token, such that
#         each sentence in the batch now has same number of words and each word has an equal
#         number of characters
#         Output shape: (batch_size, max_sentence_length, max_word_length)
#     """
#     # Words longer than 21 characters should be truncated
#     max_word_length = 21

#     ### YOUR CODE HERE for part 1f
#     ### TODO:
#     ###     Perform necessary padding to the sentences in the batch similar to the pad_sents()
#     ###     method below using the padding character from the arguments. You should ensure all
#     ###     sentences have the same number of words and each word has the same number of
#     ###     characters.
#     ###     Set padding words to a `max_word_length` sized vector of padding characters.
#     ###
#     ###     You should NOT use the method `pad_sents()` below because of the way it handles
#     ###     padding and unknown words.

#     # Initialize the list for the padded sents
#     sents_padded = []

#     # Compute the max number of words in a sentence
#     max_length = max([len(sent) for sent in sents])

#     # Iterate each sentence and pad it
#     for sent in sents:
        
#         words_padded = []
#         for word in sent:
#             # If the length of the word is longer than the max_char
#             # Truncate if necessary
#             if len(sent) > max_word_length:
#                 word = word[:max_word_length]
#             else:
#                 word = word + [char_pad_token] * (max_word_length - len(word))
#             # Append the padded word into a list
#             words_padded.append(word)
#         # Assure that the number of words in a setence is unform
#         if len(words_padded) < max_length:
#             for _ in range(max_length - len(words_padded)):
#                 words_padded.append([char_pad_token] * max_word_length)
#         sents_padded.append(words_padded)

#     return sents_padded


# def pad_sents(sents, pad_token):
#     """ Pad list of sentences according to the longest sentence in the batch.
#     @param sents (list[list[str]]): list of sentences, where each sentence
#                                     is represented as a list of words
#     @param pad_token (str): padding token
#     @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
#         than the max length sentence are padded out with the pad_token, such that
#         each sentences in the batch now has equal length.
#     """
#     # Initialize the list for the padded sents
#     sents_padded = []

#     # Compute the max length
#     max_length = max([len(sent) for sent in sents])

#     # iterate each sentence and pad it
#     for sent in sents:
#         sent = sent + [pad_token] * (max_length - len(sent))
#         sents_padded.append(sent)

#     return sents_padded



# def read_corpus(file_path, source):
#     """ Read file, where each sentence is dilineated by a `\n`.
#     @param file_path (str): path to file containing corpus
#     @param source (str): "tgt" or "src" indicating whether text
#         is of the source language or target language
#     """
#     data = []
#     for line in open(file_path):
#         sent = line.strip().split(' ')
#         # only append <s> and </s> to the target sentence
#         if source == 'tgt':
#             sent = ['<s>'] + sent + ['</s>']
#         data.append(sent)

#     return data


# def batch_iter(data, batch_size, shuffle=False):
#     """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
#     @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
#     @param batch_size (int): batch size
#     @param shuffle (boolean): whether to randomly shuffle the dataset
#     """
#     batch_num = math.ceil(len(data) / batch_size)
#     index_array = list(range(len(data)))

#     if shuffle:
#         np.random.shuffle(index_array)

#     for i in range(batch_num):
#         indices = index_array[i * batch_size: (i + 1) * batch_size]
#         examples = [data[idx] for idx in indices]

#         examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
#         src_sents = [e[0] for e in examples]
#         tgt_sents = [e[1] for e in examples]

#         yield src_sents, tgt_sents


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()` 
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal 
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21 
    max_sentence_length = max([len(s) for s in sents]) # len

    """
        The elements in the input are not uniform in length nor padded. Thus, we need to pad them to make them all equal in length
        This function paddes the sentences in the batch to ensure that the sentences have the same number of words
        and each word has the same number of characters
    """

    def sent_to_vec(sent):
        """
            This function pads all the sentences and words
        """
        # List to store the padded sentences
        padded_sents_lst = []

        # We truncate the sentence whose number of words exceeds the maximum length
        for word in sent[:max_sentence_length]:
            # We also truncate the sentence whose number of characters exceeds the max length
            padded_word = word[:max_word_length]
            # pad each word until it reaches the maximum length
            padded_word += [char_pad_token] * (max_word_length - len(padded_word))
            # append the padded word into the sentence list
            padded_sents_lst.append(padded_word)
        # pad each sentence with the list of char_pad_token until it reaches the maximum sentence length 
        padded_sents_lst += [[char_pad_token] * max_word_length] * (max_sentence_length - len(padded_sents_lst))
        return padded_sents_lst
    sents_padded = [sent_to_vec(sent) for sent in sents]

    return sents_padded

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents