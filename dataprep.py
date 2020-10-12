#!/usr/bin/env python

import numpy as np
import graph_tool as gt
from graph_tool.all import *

"""
The module contains various data preparation/transformation tools.
The averages are necessary to evaluate density decay runs.
"""


def density_data_averager(data):
    """ Averages multiple density decay runs.

    The SIS and CP functions return an array containing multiple
    density decay runs with varying length. This method adds zeros
    until all runs match the longest and takes the average of those
    runs.

    Args:
        data (list[np.array]): Decay runs.

    Returns:
        np.array: Single averaged decay.
    """
    len_data = len(data)
    max_len = max([len(data[i]) for i in range(len_data)])
    dd_filled = np.zeros((len_data, max_len))
 
    for i in range(len_data):
        len_data_i = len(data[i])
        dd_filled[i] = np.concatenate((data[i], np.zeros(max_len 
                                                         - len_data_i)))
    return np.sum(dd_filled, axis=0)/len_data


def density_qs_averager(data, cut):
    """ Returns the average of multiple QS density decay runs.
    
    All QS runs have inherently the same length. The initial decay
    causes a lot of noise and can be cut in steps of 10% of total run
    time to decrease noise.

    Args:
        data (list[np.array]): QS decay runs.
        cut (int): Initial decay to cut.

    Returns:
        np.array: Single averaged QS decay. 
    """
    n = len(data)
    t = int(len(data[0])/10)
    return np.array([np.mean(data[i][len(data[0])-cut*t:]) for i in range(n)])


def graph_to_neighbor(g):
    """Returns a neighbor list from a given graph g.

    The elements are arrays containing the neighbors of the indexed
    node. The nodes are shifted by +1 to be compatible with the
    dynamics.py methods.
    
    Args:
        g (gt.Graph): Graph to transform
        
    Return:
        list (np.array): List of neighbors
    """
    adj = gt.spectral.adjacency(g)
    return [np.nonzero(row)[1]+1 for row in adj]
