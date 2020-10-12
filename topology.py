#!/usr/bin/env python

"""
Various topological network measures that are not included in
graph_tool.
"""

import numpy as np
import graph_tool as gt
import dataprep
from graph_tool.all import *


def global_efficiency_estimate(g, n):
    """Returns the global efficiency of a graph g, estimated
    from n node samples.

    Applicable to large networks.
    """
    nv = g.num_vertices()
    eff = 0
    distance = gt.stats.distance_histogram(g, samples=n)
    for i in range(1, len(distance[0]+1)):
        eff += distance[0][i]*1/distance[1][i]

    return eff/(n*(nv-1))


def global_efficiency(g):
    """Returns the global efficiency of a network.

    Too demanding for large networks.
    """
    nv = g.num_vertices()
    eff = 0
    hist = gt.stats.distance_histogram(g)
    for i in range(1, len(hist[0]+1)):
        eff += hist[0][i]*1/hist[1][i]
    
    return eff/(nv*(nv-1))


def geodesic_entropy(g, n):
    """Returns the geodesic entropy of a node n in graph g."""
    # Geodesic Probability
    dist = gt.topology.shortest_distance(g, source=n)
    dist_arr = dist.get_array()
    r = int(dist_arr.max())
    nv = len(dist_arr)
    
    # Geodesic Entropy
    s_geo = 0
    for ri in range(1, r+1):
        p_r = len(gt.util.find_vertex(g, dist, r))/(nv-1)
        s_geo += -p_r*np.log(p_r)
        
    return s_geo


def char_geodesic_entropy(g):
    """Returns the characteristic geodesic entropy of a given graph g."""
    n = g.num_vertices()
    geo_entropy = 0
    for i in range(n):
        geo_entropy += geodesic_entropy(g, i)

    return geo_entropy/n


def local_eff(g):
    """Returns the local efficiency of a graph g."""
    n = g.num_vertices()
    eff_sum = 0
    for node in range(n):
        # Extract the neighbors of node_i
        vfilt = g.new_vertex_property('bool')
        neighbor = g.vertex(node).all_neighbors()

        # Create a sub graph containing those neighbors
        # and calculate their shortest distances
        for n_node in neighbor:
            vfilt[n_node] = True
        sub = gt.GraphView(g, vfilt)
        sub = Graph(sub, prune=True)  # prune for true copy
        sub_dist = gt.topology.shortest_distance(sub)

        # Calculate the local efficiency of node_i
        eff_sum_i = 0
        for dist_row in sub_dist:
            for dist in dist_row:
                if dist != 0:
                    eff_sum_i += 1/dist

        deg = g.vertex(node).out_degree()
        if deg > 1:
            eff_sum_i = eff_sum_i/(deg*(deg - 1))
        eff_sum += eff_sum_i
    
    return eff_sum/n


def ext_local_eff(g, d):
    """Returns the extended local efficiency of a network g up to a
    neighbor depth d.

    The measure is experimental, not an established concept.
    Motivated by the concept of extended clustering.
    """
    n = g.num_vertices()
    eff_sum = 0
    # Extract the neighbors of node_i up to depth d and
    # create a sub graph containing those neighbors
    for node in range(n):
        vfilt = g.new_vertex_property('bool')
        neighbors = []
        node_n1 = node
        j = -1
        
        # Code is a bit convoluted to have variable depth
        for _ in range(d):
            k = len(neighbors)
            while j < k:
                for node_n2 in g.vertex(node_n1).all_neighbors():
                    neighbors.append(int(node_n2))
                    vfilt[node_n2] = True
                j += 1
                node_n1 = neighbors[j]

        # Calculate the neighbor distances up to depth d
        deg = g.vertex(node).out_degree()
        sub = gt.GraphView(g, vfilt)
        sub_dist = []
        for i in range(deg):
            sub_dist = sub_dist + list(gt.topology.shortest_distance(
                sub, source=g.vertex(neighbors[i]), target=[g.vertex(m) for m in neighbors[:deg]]))
        
        # Calculate the local efficiency of node_i
        eff_sum_i = 0
        for dist in sub_dist:
            if dist != 0:
                eff_sum_i += 1/dist
        
        eff_sum += eff_sum_i/(deg*(deg - 1))

    return eff_sum/n


def susceptibility(data, cut, size=10**6):
    """Returns the dynamic susceptibility via QS runs"""
    rho = dataprep.density_qs_averager(data, cut)
    return size*(dataprep.density_qs_averager(data**2, cut) - rho**2)/rho
