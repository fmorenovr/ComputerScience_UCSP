#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.files import *

def preprocessData(inFilename, outFilename, Type):
  if Type == "CSV":
    preprocessDataCSV(inFilename, outFilename)
  elif Type == "TXT":
    preprocessDataTXT(inFilename, outFilename)

def preprocessDataCSV(inFilename, outFilename):
  # load document
  in_filename = "data/" + inFilename
  obs = load_csv(in_filename)
  print(obs[:200])
  
  # clean document
  tokens = clean_csv(obs)
  print(tokens[:200])
  print('Total Tokens: %d' % len(tokens))
  print('Unique Tokens: %d' % len(set(tokens)))
  
  # organize into sequences of tokens
  length = 50 + 1
  sequences = list()
  for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
  print('Total Sequences: %d' % len(sequences))
  
  # save sequences to file
  out_filename = "data/" + outFilename
  save_doc(sequences, out_filename)

def preprocessDataTXT(inFilename, outFilename):
  # load document
  in_filename = "data/" + inFilename
  doc = load_doc(in_filename)
  print(doc[:200])
   
  # clean document
  tokens = clean_doc(doc)
  print(tokens[:200])
  print('Total Tokens: %d' % len(tokens))
  print('Unique Tokens: %d' % len(set(tokens)))
   
  # organize into sequences of tokens
  length = 50 + 1
  sequences = list()
  for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
  print('Total Sequences: %d' % len(sequences))
   
  # save sequences to file
  out_filename = "data/" + outFilename
  save_doc(sequences, out_filename)
