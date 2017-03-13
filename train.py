#! /usr/bin/env python

# 02/2017 | Moss McLaughlin

import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from GRU_text_gen import GRUTheano
#import load_sentenceCorpus as loadS

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.002"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "6000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "64"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "100"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")

INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "data/sentenceCorpusData.txt")
#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "data/imdb_Data.txt")

OUTPUT_DATA_FILE = os.environ.get("OUTPUT_DATA_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "10"))


'''
Auto-names output based on time-stamp and parameters
'''
if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

if not OUTPUT_DATA_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  OUTPUT_DATA_FILE = "../gen_data_%s.txt" % ts
# Load data
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

# Build model
model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time

t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print("SGD Step time: %f seconds" % (t2 - t1))
print("Approximate time per Epoch: %f minutes" % ((t2 - t1) / 60 * len(x_train)))
print('\n')
sys.stdout.flush()


# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(x_train[:10], y_train[:10])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)
  generate_sentences(model, 100, index_to_word, word_to_index,loss,OUTPUT_DATA_FILE)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

  
for epoch in range(NEPOCH):
  print('\n'*3)
  print('Starting Epoch %i / %i...' % (epoch,NEPOCH))
  print('--------------------------')
  train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)
