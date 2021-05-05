#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from .functions import getFileFormat, setName
import pandas as pd

pp_dir = "data/scores/placepulse/"

def getScores(metadata, value_to_search=""):
  dataset_name = metadata[0]
  if dataset_name == "pp1" or dataset_name == "placepulse_1":
    metadata[0] = pp_dir+"pp1/Qscores/"
    results = PlacePulse(metadata, value_to_search)
  if dataset_name == "pp2" or dataset_name == "placepulse_2":
    metadata[0] = pp_dir+"pp2/Qscores/"
    results = PlacePulse(metadata, value_to_search)
  else:
    metadata[0] = pp_dir+"pp1/Qscores/"
    results = PlacePulse(metadata, value_to_search)
  return results

def getAggregateStatistics(dataset_name="pp1"):
  if dataset_name == "pp1" or dataset_name == "placepulse_1":
    path = pp_dir+"pp1/Qscores/"
    results = pd.read_csv(path+"aggregate_statistics.csv")
  if dataset_name == "pp2" or dataset_name == "placepulse_2":
    path = pp_dir+"pp2/Qscores/"
    results = pd.read_csv(path+"aggregate_statistics.csv")
  else:
    path = pp_dir+"pp1/Qscores/"
    results = pd.read_csv(path+"aggregate_statistics.csv")
  return results
  
# Place Pulse 1.0
def PlacePulse(metadata, value_to_search=""):
  path, city, metric = metadata[0], metadata[1], metadata[2]
  #print("Preparing scores ...")
  result = pd.read_csv(path+city+".csv")
  result.sort_values(metric, ascending=False, inplace=True)
  #print(result)
  if value_to_search=="":
    return [result['ID'].values, result[metric].values, result['Lat'].values, result['Lon'].values]
  else:
    row_search = result[result['ID']==int(value_to_search)]
    return [row_search['ID'].values, row_search[metric].values, row_search['Lat'].values, row_search['Lon'].values]

def placePulse_1(metadata, value_to_search=""):
  path, city, metric = metadata[0], metadata[1], metadata[2]
  typefile = getFileFormat(path)
  print("Preparing scores ...")
  if typefile == "json":
    result = pd.read_json(path)
  elif typefile == "csv":
    result = pd.read_csv(path)
  # Rename to Qs, Qu, Qw
  result.rename(columns={'QS Upperclass': 'Wealth', 'Error in QS Upperclass': 'Wealth_Err', 'QS Unique': 'Unique', 'Error in QS Unique': 'Unique_Err', 'QS Safer': 'Safe', 'Error in QS Safer': 'Safe_Err', 'File_Location': 'Filename'}, inplace=True)
  # deleting some columns
  result.drop(columns=["Safe_Err", "Unique_Err", "Wealth_Err"], inplace=True)
  values = result[metric].values
  result.loc[:, metric] = pd.to_numeric(values, errors='coerce')
  #set mean to NULL values
  result[metric].fillna(result[metric].mean(), inplace=True)
  # sorting by image name
  result.sort_values(metric, ascending=False, inplace=True)
  # normalized values
  #result['Y'] = preprocessing.normalize([result[metric].values])[0]
  
  #result['FilenameSort'] = result['Filename'].str.extract('(\d+)', expand=False).astype(int)
  #result.sort_values('FilenameSort', inplace=True, ascending=True)
  #result.drop('FilenameSort', axis=1, inplace=True)
  
  result['Filename'] = result['Filename'].apply(setName)

  query = result[result['City'].str.match(city)]
  #normal_scores = query['Y'].values
  print("Scores and images Done")
  
  if value_to_search=="":
    return [query['Filename'].values, query[metric].values, query['Lat'].values, query['Lon'].values, city, metric]
  else:
    row_search = query[query['Filename']==value_to_search]
    return [row_search['Filename'].values, row_search[metric].values, row_search['Lat'].values, row_search['Lon'].values, city, metric]
