#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import array
from random import randint
from pickle import dump, load

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils.files import load_doc
from utils.preprocess import preprocessData

EPOCHS = 1000
BATCH_SIZE = 256

# generate a sequence from a language model
def generate_sequence(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # predict probabilities for each word
    yhat = model.predict_classes(encoded, verbose=0)
    # map predicted word index to word
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    # append to input
    in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

def trainModel(rawdata, trainfilename, fileType, modelType):
  print("\n============================================")
  print("             Preprocessing data               ")
  print("============================================\n")

  preprocessData(rawdata, trainfilename, fileType)
  # load
  in_filename = "data/" + trainfilename
  doc = load_doc(in_filename)
  lines = doc.split('\n')

  # integer encode sequences of words
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  sequences = tokenizer.texts_to_sequences(lines)
  # vocabulary size
  vocab_size = len(tokenizer.word_index) + 1

  # separate into input and output
  sequences = array(sequences)
  X, y = sequences[:,:-1], sequences[:,-1]
  y = to_categorical(y, num_classes=vocab_size)
  seq_length = X.shape[1]

  # define model
  model = Sequential()
  if modelType == 1:
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
  elif modelType == 2:
    model = Sequential()
    model.add(LSTM(256, input_shape=seq_length))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
  print(model.summary())
  # compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # define the checkpoint
  filepath="checkpoints/weights-{epoch:02d}-{loss:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]
  
  print("\n============================================")
  print("                 Training Data                ")
  print("============================================\n")
  
  # fit model
  model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
  # save the model to file
  model.save('models/model.h5')
  # save the tokenizer
  dump(tokenizer, open('tokens/tokenizer.pkl', 'wb'))

def testModel(filename, modelname):
  # load cleaned text sequences
  in_filename = "data/" + filename
  doc = load_doc(in_filename)
  lines = doc.split('\n')
  seq_length = len(lines[0].split()) - 1
   
  # load the model
  model = load_model(modelname)
  
  # load the tokenizer
  tokenizer = load(open('tokens/tokenizer.pkl', 'rb'))
  
  print("\n============================================")
  print("                  Prediction                  ")
  print("============================================\n")
  # select a seed text
  while(1):

    seed_text = str(input("Write a sentence to predict: "))
    
    if seed_text == "owari" :
      print("\n\nEnd Test\n")
      break
    
    numWords = input("Write number of words: ")
    if len(numWords) == 0:
      print("\nWrite number of words to predict !!\n")
      continue
    else:
      # generate new text
      generated = generate_sequence(model, tokenizer, seq_length, seed_text, int(numWords))
      print("\nGenerated Text: " + seed_text + " ... " + generated+"\n")
