#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from pandas import read_csv

# load csv into memory
def load_csv(filename):
  df = read_csv(filename)
  obs = df['observations']
  return obs

# turn a csv into clean tokens
def clean_csv(doc_csv):
  # split into tokens with white space
  tokens = []
  for w in doc_csv:
    tokens = tokens + w.split()
  # remove punctuation from each token
  table = str.maketrans('', '', string.punctuation)
  # remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
  # make lower case
  tokens = [word.lower() for word in tokens]
  return tokens

# load doc into memory
def load_doc(filename):
  # open the file as read only
  filen = open(filename, 'r', encoding='utf-8')
  # read all text
  text = filen.read()
  # close the file
  filen.close()
  return text

# turn a doc into clean tokens
def clean_doc(doc):
  # replace '--' with a space ' '
  doc = doc.replace('--', ' ')
  # split into tokens by white space
  tokens = doc.split()
  # remove punctuation from each token
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
  # make lower case
  tokens = [word.lower() for word in tokens]
  return tokens
 
# save tokens to file, one dialog per line
def save_doc(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()
