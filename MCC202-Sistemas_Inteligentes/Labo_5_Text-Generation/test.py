#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.model import testModel

dataTestTXT = "data_seq_txt.txt"
dataTestCSV = "data_seq_csv.txt"
dataTest = dataTestCSV
modelName = "models/model.h5"
#modelName = "checkpoints/weights-01-6.9883.hdf5"

def main():
  testModel(dataTest, modelName)

if __name__ == "__main__":
  main()
