#!/usr/python

from random import *
file = open("data.txt","w") 

for i in range(120000000):
  file.write(str(random())+"\n")
 
file.close()
