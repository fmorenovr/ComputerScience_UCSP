#!/usr/bin/env python

from math import log, sqrt

def comparison(n,c):
  #print 1.0/n, 1, log(log(n,2),2), log(n,2), n**(1.0/3.0), (log(n,2))**2, n/log(n,2), n, n**(1.00001), n**(1.5), n**2/log(n,2), n**2;
  #return 1.0/n < 1 < log(log(n,2),2) <log(n,2)< n**(1.0/3.0)<log(n,2)**2< n/log(n,2)< n < n**(1.00001) < n**(1.5) < n**2/log(n,2)  < n**2
  #return n**2<n**2+n**(5.0/2.0)<n**(log(n,2)) <2**n < 5**n < log(n,2)**n < n**n < n**(n**2)
  return c*log(n,2) < 3*log(n,2) + log(log(n,2),2)

for i in range(3,100000):
  if comparison(i,3) == False:
      print "false", i;
