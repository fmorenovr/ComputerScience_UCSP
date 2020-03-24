#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.model import trainModel

dataRawCSV = "data_csv.csv"
dataRawTXT = "data_txt.txt"
dataRaw = dataRawCSV

dataTrainTXT = "data_seq_txt.txt"
dataTrainCSV = "data_seq_csv.txt"
dataTrain = dataTrainCSV

dataType = "CSV"

def main():
  trainModel(dataRaw, dataTrain, dataType, 1)

if __name__ == "__main__":
  main()
