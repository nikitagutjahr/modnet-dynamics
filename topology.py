#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import graph_tool as gt
from graph_tool.all import *

# Various topological network measures.
# The graphs must be in graph_tool format.

# Estimates the global efficiency of a given graph g
# with N nodes via n sample nodes. More efficienct for
# large neworks.
def global_efficiency_estimate(g, N, n):
    
    eff = 0
    distance = gt.stats.distance_histogram(g, samples=n)
    for i in range(1, len(distance[0]+1)):
        eff += distance[0][i]*1/distance[1][i]
    
    return eff/(n*(N-1))


# Calculates the global efficiency of a given graph g
# with N nodes. To demanding for large networks.
def global_efficiency(g, N):
    
    eff = 0
    hist = gt.stats.distance_histogram(g)
    for i in range(1, len(hist[0]+1)):
        eff += hist[0][i]*1/hist[1][i]
    
    return eff/(N*(N-1))


# Calculates the geodesic entropy of a given graph g
# via n samples.
def geodesic_entropy(g, n):
    
    # Geodesic Probability
    dist = gt.topology.shortest_distance(g, source=n)
    dist_arr = dist.get_array()
    R = int(dist_arr.max())
    N = len(dist_arr)
    
    # Geodesic Entropy
    s_geo = 0
    for r in range(1, R+1):
        p_r = len(gt.util.find_vertex(g_mmn, dist, r))/(N-1)
        s_geo += -p_r*np.log(p_r)
        
    return s_geo

