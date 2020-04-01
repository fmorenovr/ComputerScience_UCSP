#!/usr/bin/python

import os
import pandas
import numpy
import matplotlib.pyplot as plt

scores_dir = "Qscores"

def setName(name):
  aux = str.split(name, "id")
  aux1 = str.split(aux[1], "_")
  name = aux1[1]
  return name

def verifyDir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def preprocess_pp1(path):
  result = pandas.read_json(path)
  #result.replace('#VALUE!', numpy.nan, inplace=True)
  result.rename(columns={'QS Upperclass': 'wealthy', 'Error in QS Upperclass': 'wealthy_Err', 'QS Unique': 'uniquely', 'Error in QS Unique': 'uniquely_Err', 'QS Safer': 'safety', 'Error in QS Safer': 'safety_Err', 'File_Location': 'Filename'}, inplace=True)
  print(result)
  
  cols = list(result)
  cols = [cols[7], cols[11], cols[3], cols[10], cols[2], cols[4], cols[6], cols[9], cols[12], cols[8], cols[5], cols[0], cols[1]]
  
  #print(list(result))
  result = result.loc[:,cols]
  result['Filename'] = result['Filename'].apply(setName)
  #result.drop(columns=["Filename"], inplace=True)
  result.drop(columns=["ID"], inplace=True)
  result.rename(columns={'Filename':'ID'}, inplace=True)
  #print(result)
  
  cities = list(dict.fromkeys(result['City'].values))
  print(cities)
  metrics = [cols[7], cols[9], cols[11]]
  numImg = []
  sigmas = {metric: [] for metric in metrics}
  means = {metric: [] for metric in metrics}
  print(metrics)
  
  for city in cities:
    query = result[result['City'].str.match(city)]
    query.drop(columns=["City"], inplace=True)
    print(query)
    
    num_img = len(query['ID'].values)
    numImg.append(num_img)
    
    for i, metric in enumerate(metrics):
      values = query[metric].values
      query.loc[:, metric] = pandas.to_numeric(values, errors='coerce')
      query[metric].fillna(query[metric].mean(), inplace=True)
      sigmas[metric].append(query[metric].std())
      means[metric].append(query[metric].mean())

    query.to_csv(scores_dir+"/"+city+".csv", index=False)
    
  #print(sigmas)
  #print(means)
  #print(numImg)
  
  data = {"city": cities,
          "numImages": numImg,
          "safety_mean": means['safety'],
          "safety_sigma": sigmas['safety'],
          "wealthy_mean": means['wealthy'],
          "wealthy_sigma": sigmas['wealthy'],
          "uniquely_mean": means['uniquely'],
          "uniquely_sigma": sigmas['uniquely'],
  }

  aggregate_statistics = pandas.DataFrame(data, columns= ['city', 'numImages', "safety_mean", "safety_sigma", "wealthy_mean", "wealthy_sigma", "uniquely_mean", "uniquely_sigma"])

  aggregate_statistics.to_csv(scores_dir+"/"+"aggregate_statistics.csv", index=False)
  
  plt.legend(title="Num Images per city - Place Pulse 1.0")
  plt.bar(cities, numImg)
  plt.tick_params(axis="x", rotation=0, labelsize=10)
  plt.savefig(scores_dir+"/"+"cities_pp1.png", bbox_inches='tight', pad_inches = 0.5)
  plt.clf()
  plt.cla()
  plt.close()

if __name__ == '__main__':
  verifyDir(scores_dir)
  path = "placepulse_1.json"
  preprocess_pp1(path)
