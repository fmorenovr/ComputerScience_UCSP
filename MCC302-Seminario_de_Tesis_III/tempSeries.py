import json
import pandas as pd
import matplotlib.pyplot as plt

import os

def verifyDir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

tempSeries_dir = "outputs/tempSeries/"
verifyDir(tempSeries_dir)

with open('tempSeries.json') as json_file:
  data = json.load(json_file)

for node in data:
  cid = str(node['cid'])
  months = [i['month'] for i in node['serie']]
  values = [i['value'] for i in node['serie']]

  plt.plot(months, values)
  #plt.savefig(cid+'/ts.png')
  plt.savefig(tempSeries_dir+cid+'.png')
  plt.clf()
  plt.cla()
  plt.close()
