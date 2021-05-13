#!/usr/bin/python

# -*- coding: utf-8 -*-
# encoding: utf-8

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import zipfile
#import geocoder
from unicodedata import normalize

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from utils import verifyFile, verifyDir
from .variables import country_city_dataset, scores_dir, ROOT_DIR, summary_dir
from utils.datasets import newRange, read_zip_file

index_to_text_preprocess=-1

# Get number of wins, draws, and losses for each image in a specific position (right or left) and get the id of wins and losses images against
def get_WDL_Info(result, value_to_search, position, category):
  row = result[(result[position+'_id']==value_to_search) & (result['category']==category)]
  #print(row)
  if len(row) == 0:
    return [], 0, 0, 0, [], []
  else:
    wins = row[row['winner']==position]
    draws = row[row['winner']=="equal"]
    losses = row[(row['winner']!="equal") & (row['winner']!=position)]
    return row, len(wins), len(draws), len(losses), wins, losses

# Get all ids of W, L, wins, draws and losses from images in each category
def getSummaryScores(result, value_to_search, category):
  row_l, w_l, d_l, l_l, wins_l, losses_l = get_WDL_Info(result, value_to_search, "left", category)
  row_r, w_r, d_r, l_r, wins_r, losses_r = get_WDL_Info(result, value_to_search, "right", category)
  
  # verificando len y obteniendo Lat, Lon
  if len(row_l) != 0:
    lat = list(dict.fromkeys(row_l['left_lat'].values))[0]
    lon = list(dict.fromkeys(row_l['left_long'].values))[0]
  elif len(row_r) != 0:
    lat = list(dict.fromkeys(row_r['right_lat'].values))[0]
    lon = list(dict.fromkeys(row_r['right_long'].values))[0]
  else:
    return []
  
  # Array de Id de las images a las que vencio
  wins_id = []
  # Array de Id de las images con las que perdio
  losses_id = []
  if len(wins_r)!=0:
    aux = list(dict.fromkeys(wins_r['left_id'].values))
    wins_id = wins_id + aux
  if len(losses_r)!=0:
    aux = list(dict.fromkeys(losses_r['left_id'].values))
    losses_id = losses_id + aux
    
  if len(wins_l)!=0:
    aux = list(dict.fromkeys(wins_l['right_id'].values))
    wins_id = wins_id + aux
  if len(losses_l)!=0:
    aux = list(dict.fromkeys(losses_l['right_id'].values))
    losses_id = losses_id + aux
  
  w = w_l + w_r
  d = d_l + d_r
  l = l_l + l_r
  
  total = w + d + l
  if total == 0:
    return []
  
  W = w*1.0 / (total)
  L = l*1.0 / (total)
  
  return [W, L, w, d, l, lat, lon, list(dict.fromkeys(wins_id)), list(dict.fromkeys(losses_id))]

# Calculate QScore per category
def calculateQscore(result, value_to_search, wins_id, losses_id):
  W_, L_ = 0.0, 0.0
  if len(wins_id) != 0:
    for w_id in wins_id:
      w_id_info_score = getInfoScore(result, w_id)
      if w_id_info_score:
        w_, _, _, _, _, _ = w_id_info_score
        W_ = W_ + w_
  if len(losses_id) != 0:
    for l_id in losses_id:
      l_id_info_score = getInfoScore(result, l_id)
      if l_id_info_score:
        _, l_, _, _, _, _ = l_id_info_score
        L_ = L_ + l_

  image_info_score = getInfoScore(result, value_to_search)
  if image_info_score:
    W, _, _, _, num_wins, num_losses = image_info_score
    q = W + 1.0
    if num_wins!=0:
      q = q + 1.0/num_wins * W_
    if num_losses!=0:
      q = q - 1.0/num_losses * L_
    Q = 10.0/3*(q)
  else:
    Q=0.0
  return Q

# Get info of each row
def getInfoScore(result, value_to_search):
  query = result[result["ID"]==value_to_search]
  
  if len(query)!=0:
    return query["W_ratio"].values[0], query["L_ratio"].values[0], query["wins"].values[0], query["losses"].values[0], query["num_wins"].values[0], query["num_losses"].values[0]
  else:
    return None

# GetInfoSummaryRow
def GetInfoSummaryRow(result, value_to_search):
  query = result[result["ID"]==value_to_search]
  return query

## PART 0: get information
def preprocess_data(data, filename):
  result = pandas.read_csv(data.open(filename))
  #print(result)

  # ids images
  left_ids = list(dict.fromkeys(result['left_id'].values))
  right_ids = list(dict.fromkeys(result['right_id'].values))
  total_ids = left_ids + right_ids
  img_ids = list(dict.fromkeys(total_ids))

  # just to verify that an image does not compare with itself
  row_equals = result[result['left_id'] == result['right_id']]
  if len(row_equals)>0:
    print("Length of equal comparisons:", len(row_equals))
    result = result[result['left_id'] != result['right_id']]

  # Categories
  metrics = list(dict.fromkeys(result['category'].values))

  print("Number of Images: {} to evaluate from a total of: {} comparisons in {} perception categories.".format(len(img_ids), len(result), len(metrics)))
  
  return result, img_ids, metrics

## PART 1: Preprocess Lat, Lon, W, L, wins, draws, and losses for each image
def preprocess_ratios(data, filename, save_file=False, summary_dir=ROOT_DIR+"pp2/Summaries/"):
  verifyDir(summary_dir)
  print("Preparing files to summarize all wins, losses, and draw for all images per category ...")
  
  result, img_ids, metrics = preprocess_data(data, filename)

  for category in metrics:
    print("Category {}:".format(category))
    
    IDs = []
    W = []
    L = []
    w = []
    d = []
    l = []
    num_wins = []
    num_losses = []
    lat = []
    lon = []

    if verifyFile(summary_dir+"summary_scores_"+category+".csv"):
      continue
    
    for value_to_search in img_ids[:index_to_text_preprocess]:
      imgPreData = getSummaryScores(result, value_to_search, category)
      if len(imgPreData) == 0:
        continue
      [W_, L_, w_, d_, l_, lat_, lon_, wins_id, losses_id] = imgPreData
      
      IDs.append(value_to_search)
      W.append(W_)
      L.append(L_)
      w.append(w_)
      d.append(d_)
      l.append(l_)
      num_wins.append(len(wins_id))
      num_losses.append(len(losses_id))
      lat.append(lat_)
      lon.append(lon_)
      print("- Image-ID: {}, Wp: {:.3f}, Lp: {:.3f}, Wins: {}, Draws: {}, Losses: {}, Num Wins: {}, Num Losses: {}".format(value_to_search, W_, L_, w_, d_, l_, len(wins_id), len(losses_id)))
      
    data = {"ID": IDs,
            "Lat": lat,
            "Lon": lon,
            "W_ratio": W,
            "L_ratio": L,
            "wins": w,
            "draws": d,
            "losses": l,
            "num_wins": num_wins,
            "num_losses": num_losses,
    }
    
    scores = pandas.DataFrame(data, columns= ['ID', 'Lat', 'Lon', 'W_ratio', "L_ratio", "wins", "draws", "losses", "num_wins", "num_losses"])
    
    if save_file:
      scores.to_csv(summary_dir+"summary_scores_"+category+".csv", index=False)

    print("Done")

  return

## PART 2: Constructing real ponderated scores for each image in each caegory
def preprocess_Qscores(data, filename, save_file=False, summary_dir=ROOT_DIR+"pp2/Summaries/"):
  verifyDir(summary_dir)
  print("Preparing files to calculate Qscore per category for all images ...")
  
  result, img_ids, metrics = preprocess_data(data, filename)
  
  for category in metrics:
  
    summary = pandas.read_csv(summary_dir+"summary_scores_"+category+".csv")
    print("Category {} with {} images".format(category, len(summary)))
    
    if 'Q' in summary.columns.values:
      continue
    
    Qscores = []
    
    for value_to_search in img_ids[:index_to_text_preprocess]:
      imgPreData = getSummaryScores(result, value_to_search, category)
      if len(imgPreData) == 0:
        continue
      [W_, L_, w_, d_, l_, lat_, lon_, wins_id, losses_id] = imgPreData
      
      Q = calculateQscore(summary, value_to_search, wins_id, losses_id)
      Qscores.append(Q)
      print("- Image-ID: {}, QScore {:.3f}".format(value_to_search, Q))
    
    summary['Q'] = Qscores
    
    if save_file:
      summary.to_csv(summary_dir+"summary_scores_"+category+".csv", index=False)
      
    print("Done")
    
  return

## PART 3: Summarize all categories in all images
def preprocess_images(data, filename, save_file=False, summary_dir=ROOT_DIR+"pp2/Summaries/"):
  verifyDir(summary_dir)
  print("Preparing files to summarize image and Qscores in all categories ...")
  
  if verifyFile(summary_dir+"summary_scores.csv"):
    return
  
  result, img_ids, metrics = preprocess_data(data, filename)
  
  IDs = []
  lat = []
  lon = []
  Qscores = {metric: [] for metric in metrics}
  lat_lon = False
  City = []
  Country = []
  Continent = []
  
  i = 0
  
  for value_to_search in img_ids[:index_to_text_preprocess]:
  
    IDs.append(value_to_search)
  
    for category in metrics:
      summary = pandas.read_csv(summary_dir+"summary_scores_"+category+".csv")
      query = GetInfoSummaryRow(summary, value_to_search)
      
      if len(query)!=0:
        Qscores[category].append(query['Q'].values[0])
        if lat_lon == False:
          lat.append(query['Lat'].values[0])
          lon.append(query['Lon'].values[0])
          lat_lon = True
      else:
        Qscores[category].append(None)

    lat_lon = False
    
    for continent_obj in country_city_dataset:
      pol_cont = Polygon(continent_obj["gps"])
      inside_cont =  Point(lat[i], lon[i]).within(pol_cont)
      if inside_cont:
        for city_country in continent_obj['cities']:
          pol_count = Polygon(city_country["gps"])
          inside_count = Point(lat[i], lon[i]).within(pol_count)
          if inside_count:
            continent = continent_obj['continent']
            country = city_country["country"]
            city = city_country["city"]
            Continent.append(continent)
            Country.append(country)
            City.append(city)
            break
        break
    
    print("- Image-ID: {}, Lat: {:.6}, Lon: {:.6}, Continent: {}, Country: {}, City: {}, Qwealthy {}, Qdepressing {}, Qsafety {}, Qlively {}, Qboring {}, Qbeautiful {}".format(value_to_search, lat[i], lon[i], continent, country, city, Qscores['wealthy'][i], Qscores['depressing'][i], Qscores['safety'][i], Qscores['lively'][i], Qscores['boring'][i], Qscores['beautiful'][i]))
    
    i = i + 1
    
  data = {"Continent": Continent,
          "Country": Country,
          "City": City,
          "img_id": IDs,
          "ID": range(1,len(IDs)+1),
          "Lat": lat,
          "Lon": lon,
          "wealthy": Qscores['wealthy'],
          "depressing": Qscores['depressing'],
          "safety": Qscores['safety'],
          "lively": Qscores['lively'],
          "boring": Qscores['boring'],
          "beautiful": Qscores['beautiful'],
  }
  
  scores = pandas.DataFrame(data, columns= ["Continent", "Country", "City", 'ID', 'img_id', 'Lat', 'Lon', 'wealthy', 'depressing', 'safety', 'lively', 'boring', 'beautiful'])
  
  if save_file:
    scores.to_csv(summary_dir+"summary_scores.csv", index=False)

  print("Done")

  return

# PART 4
def preprocess_statistics(save_file=False, scores_dir=ROOT_DIR+"pp2/Qscores/", summary_dir=ROOT_DIR+"pp2/Summaries/"):
  verifyDir(scores_dir)
  verifyDir(summary_dir)
  result = pandas.read_csv(summary_dir+"summary_scores.csv")
  #print(result)
  
  cities = list(dict.fromkeys(result['City'].values))
  #print(cities)
  cols = list(result)
  metrics = [cols[7], cols[8], cols[9], cols[10], cols[11], cols[12]]
  numImg = []
  sigmas = {metric: [] for metric in metrics}
  means = {metric: [] for metric in metrics}
  #print(metrics)
  
  for city in cities:
    query = result[result['City'].str.match(city)]
    query.drop(columns=["City"], inplace=True)
    #print(query)
    
    num_img = len(query['ID'].values)
    numImg.append(num_img)
    
    for i, metric in enumerate(metrics):
      values = query[metric].values
      query.loc[:, metric] = pandas.to_numeric(values, errors='coerce')
      query[metric].fillna(query[metric].mean(), inplace=True)
      sigmas[metric].append(query[metric].std())
      means[metric].append(query[metric].mean())

    query.to_csv(scores_dir+city+".csv", index=False)
    
  #print(sigmas)
  #print(means)
  #print(numImg)
  
  data = {"city": cities,
          "numImages": numImg,
          "safety_mean": means['safety'],
          "safety_sigma": sigmas['safety'],
          "wealthy_mean": means['wealthy'],
          "wealthy_sigma": sigmas['wealthy'],
          "lively_mean": means['lively'],
          "lively_sigma": sigmas['lively'],
          "depressing_mean": means['depressing'],
          "depressing_sigma": sigmas['depressing'],
          "boring_mean": means['boring'],
          "boring_sigma": sigmas['boring'],
          "beautiful_mean": means['beautiful'],
          "beautiful_sigma": sigmas['beautiful'],
  }

  aggregate_statistics = pandas.DataFrame(data, columns= ['city', 'numImages', "safety_mean", "safety_sigma", "wealthy_mean", "wealthy_sigma", "lively_mean", "lively_sigma", "depressing_mean", "depressing_sigma", "boring_mean", "boring_sigma", "beautiful_mean", "beautiful_sigma"])

  if save_file:
    aggregate_statistics.to_csv(scores_dir+"aggregate_statistics.csv", index=False)
  
  plt.figure(figsize=(15,5))
  plt.legend(title="Num Images per city - Place Pulse 2.0")
  plt.bar(cities, numImg)
  plt.tick_params(axis="x", which='major', rotation=90, labelsize=10)
  plt.savefig(scores_dir+"cities_pp2.png", bbox_inches='tight', pad_inches = 0.5)
  plt.clf()
  plt.cla()
  plt.close()

# PART 5
def preprocess_charts(scores_dir=ROOT_DIR+"pp2/Qscores/"):
  verifyDir(scores_dir)
  summary = pandas.read_csv(scores_dir+"aggregate_statistics.csv")
  metrics = ["safety", "lively", "depressing", "beautiful", "boring", "wealthy"]

  city = summary['city'].values
  numImages = summary['numImages'].values

  for metric in metrics:
    scores = summary[metric+'_mean'].values

    range_val = 0.3

    metric_min = scores.min()
    metric_max = scores.max()

    xmin = metric_min - range_val
    xmax = metric_max + range_val

    # Preparing range for the lines behind of each circle
    distance = newRange(numImages, numImages.min(), numImages.max(), 1, 8)
    signs = np.resize([-1, 1], int(len(distance)))
    distance = np.multiply(distance, signs)

    levels = np.tile(distance, int(np.ceil(len(numImages)/len(distance))))[:len(numImages)]
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

    # Preparing the main chart - scatter
    fig, ax = plt.subplots(figsize=[15,5])
    # Marker size in units of points^2
    volume = np.sqrt(numImages)*4
    ax.scatter(scores, np.zeros(len(scores)), s=volume, alpha=0.3, color="red")
    ax.plot([xmin + range_val, xmax - range_val], [0, 0], linewidth=1, markersize=12, color='black')

    # adding cities and lines
    for d, l, r, va in zip(scores, levels, city, vert):
      ax.annotate(r, xy=(d, l+np.sign(l)*0.5), va=va, ha="center", fontsize=8)
      ax.plot([d, d], [0, l], linewidth=1, markersize=12, color='gray')

    plt.text(xmin+0.1, 0, "Less "+metric, fontsize=10)
    plt.text(xmax-0.1, 0, "More "+metric, fontsize=10)

    #plt.xticks(np.arange(xmin, xmax, step=0.05))
    ax.axes.get_xaxis().set_visible(False)
    #plt.yticks(np.arange(-5, 5, step=0.5))
    ax.axes.get_yaxis().set_visible(False)

    ax.axes.set_xlim([xmin, xmax])
    ax.axes.set_ylim([10,-10])

    for spine in ["left", "top", "right", "bottom"]:
      ax.spines[spine].set_visible(False)

    ax.set_xlabel("Scores", fontsize=10)
    ax.set_ylabel("Names", fontsize=15)
    ax.set_title('Category: ' + metric)

    ax.grid(False)
    ax.margins(y=0.1)
    fig.tight_layout()
    #plt.show()
    plt.savefig(scores_dir+"pp2_"+metric+".png", bbox_inches='tight', pad_inches = 0.5)
    plt.clf()
    plt.cla()
    plt.close()

# PART 6
def preprocess_summary(data, filename, save_file=False, summary_dir=ROOT_DIR+"pp2/Summaries/"):
  verifyDir(summary_dir)
  
  result, img_ids, metrics = preprocess_data(data, filename)
  
  category_comparisons = []
  category_avg = []
  
  print("Number of pairwise comparisons per category:")
  total = 0
  for metric in metrics:
    row_equals = result[result['category'] == metric]
    print(metric, len(row_equals))
    total = total + len(row_equals)
    category_comparisons.append(len(row_equals))
  print("Total", total)
  
  print("\nAverage per category")
  summary = pandas.read_csv(summary_dir+"summary_scores.csv")
  metrics = ["safety", "lively", "depressing", "beautiful", "boring", "wealthy"]
  for metric in metrics:
    print(metric, summary[metric].mean())
    category_avg.append(summary[metric].mean())

  total_cities = 0
  total_imgs = 0
  print("\nNumber of cities and images per continent:")
  for continent in country_city_dataset:
    aux = summary[summary["Continent"] == continent["continent"]]
    print(continent["continent"], len(continent["cities"]), len(aux))
    total_cities = total_cities +  len(continent["cities"])
    total_imgs = total_imgs + len(aux)
  
  print("Total", total_cities, total_imgs)

def preprocess_pp2(file_name="placepulse_2.zip", filename="placepulse_2.csv"):
  path = ROOT_DIR + "pp2/"+file_name
  data = read_zip_file(path)
  # PART 1
  preprocess_ratios(data, filename, save_file=True)
  # PART 2
  preprocess_Qscores(data, filename, save_file=True)
  # PART 3
  preprocess_images(data, filename, save_file=True)
  # PART 4
  preprocess_statistics(save_file=True)
  # PART 5
  preprocess_charts()
  # PART 6
  preprocess_summary(data, filename)
