#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import graph_tool as gt
from graph_tool.all import *

# Data preperation tools.


# Averages multiple activity density arrays. Data is given as an array
# that can containts density decay runs of varying length.
def density_data_averager(data):
    len_data = len(data)
    max_len = max([len(data[i]) for i in range(len_data)])
    density_decay_filled = np.zeros((len_data, max_len))
    
    # Adds zeros until all runs match the longest run.
    for i in range(len_data):
        len_data_i = len(data[i])
        density_decay_filled[i] = np.concatenate((data[i], np.zeros(max_len - len_data_i)))
    
    return np.sum(density_decay_filled, axis=0)/len_data


# Averages quasistationary runs, all runs have inherently the same length.
# Can cut parts of the first time steps in steps of 10% to decrease
# noise.
def density_qs_averager(data, cut):
    n = len(data)
    t = int(len(data[0])/10)
    return np.array([np.mean(data[i][len(data[0])-cut*t:]) for i in range(n)])


# Returns a neighbor array from a given graph g, which can be 
# used in the spreading processes.
# The elements are lists containing the neighbors of the
# indexed node. The nodes are shifted by +1 because of details
# in the dynamics methods.
def graph_to_neighbor(g):
    adj = gt.spectral.adjacency(g)
    return np.array([np.nonzero(row) for row in adj])[:, 1]+1

