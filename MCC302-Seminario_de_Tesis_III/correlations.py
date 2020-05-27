#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from scipy import stats
from scipy.stats import pearsonr, linregress
import numpy as np
from sklearn.metrics import r2_score
x = np.random.random(10)
y = np.random.random(10)
slope, intercept, r_value, p_value, std_err = linregress(x,y)

print("r_value", r_value)
print("r_value^2", r_value**2)

correlation_matrix = np.corrcoef(x, y)
correlation_xy = correlation_matrix[0,1]
r_2 = correlation_xy**2
print("\ncorr_matrix", correlation_xy)
print("corr_matrix^2", r_2)

print("\nr2_score", r2_score(x,y))

p = pearsonr(x,y)[0]
print("\np", p)
print("p^2", p**2)
