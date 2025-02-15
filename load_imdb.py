# 02/2017 | Moss McLaughlin

import csv
import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array
from datetime import datetime
from GRU_text_gen import GRUTheano

SENTENCE_START_TOKEN = "sentence_start"
SENTENCE_END_TOKEN = "sentence_end"
UNKNOWN_TOKEN = "<UNK>"
ARTICLE_END_TOKEN = "</ARTICLE_END>"
NUM_TOKEN = "<NUM>"

def load_data(filename="data/imdb_Data.txt", vocabulary_size=12000, min_sent_characters=10):

    word_to_index = []
    index_to_word = []

    print("Reading text file...")
    with open(filename, 'rt') as f:
        txt = f.read()
        txt = txt.split(ARTICLE_END_TOKEN)
        txt = [line.split('.') for line in txt]
        txt.pop()
        txt.pop()
        for line in txt: line.pop()
        print('Raw training data:')
        print(txt[0])
        print('\n')
        print(txt[-1])
        print('\n')

        # Filter sentences
        txt = [[s for s in review if len(s) >= min_sent_characters] for review in txt]


    # Append SENTENCE_START and SENTENCE_END
    txt = [['%s %s %s' % (SENTENCE_START_TOKEN,sent,SENTENCE_END_TOKEN) for sent in article] for article in txt]

    print("Parsed %d articles." % (len(txt)))

    # Tokenize utf-8 decoded lines 
    tokenized_sentences = [[nltk.word_tokenize(line.replace('<br /><br />',' ').lower()) for line in article] for article in txt]

    for i,article in enumerate(tokenized_sentences):
        a = []
        for sent in article: a += sent
        tokenized_sentences[i] = a

    # Filter words
    print("Filtering words...\n")
    tokenized_sentences = [[w for w in line if '\\' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '*' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '[' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if ']' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '"' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if "'" not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if "`" not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if not w==''] for line in tokenized_sentences]

    # Replace all numbers with number token.
    for i,line in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w.isnumeric()==False else NUM_TOKEN for w in line]
    
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print('\nLeast freq words in vocab: ')
    print(sorted_vocab[:25])
    print('\n')



    # Replace all words not in our vocabulary with the unknown token.
    for i,line in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in line]
    

    # Remove empty sentences
    tokenized_sentences = [s for s in tokenized_sentences if (len(s) > 1)]

    print('Filtered training data:')
    print(tokenized_sentences[0])
    print('\n')


    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train, word_to_index, index_to_word

