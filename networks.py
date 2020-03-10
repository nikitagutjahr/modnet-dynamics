#!/usr/bin/env python

import random
import numpy as np
import scipy as sp
import copy
import graph_tool as gt
from graph_tool.all import *


# Distributions
def exp_dist(k, a):
    return a*np.exp(-np.asarray(k)/a)


def power_dist(k, x):
    C = sp.special.zeta(x)
    return 1/(np.asarray(k)**x*C)


# Inverse transform sampling:
# Returns a list with n nodes as indices and elements as their degrees with a min./max.
# degree of kmin/kmax. from a power law distr. with power g.
def inverse_transform_sampling_correct(dist, g, kmin, kmax, n):
    
    # Calculates the probility to find nodes with degree up to kmax and
    # the corresponding cumulative distribution.
    dist_n = dist(range(kmin, kmax+1), g)
    cumsum = np.cumsum(dist_n)
    rand = np.random.rand(n) * cumsum[-1]
    node_dist = np.array([kmin]*n)
    
    # Checks in which region of the cum. distr. the random values lie and assigns
    # the corresponding node degree.
    for i in range(n):
        j = 0
        while (cumsum[j] - rand[i]) < 0:
            node_dist[i] += 1
            j += 1
    
    return node_dist.astype(int)


# Allows to offset the deg. dist. by substracting offset degrees from the nodes,
# still keeping the min. degree in mind.
def inverse_transform_sampling_off(dist, g, kmin, kmax, n, offset):
    
    # Calculates the probility to find nodes with degree up to kmax and
    # the corresponding cumulative distribution.
    dist_n = dist(range(kmin, kmax+1), g)
    cumsum = np.cumsum(dist_n)
    rand = np.random.rand(n) * cumsum[-1]
    node_dist = np.array([kmin]*n)
    
    # Checks in which region of the cum. distr. the random values lie and assigns
    # the corresponding node degree.
    for i in range(n):
        j = offset
        while (cumsum[j] - rand[i]) < 0:
            node_dist[i] += 1
            j += 1
    
    return node_dist.astype(int)


# Creates a list of edges from a degree sequence via the construction model
# and returns a generated graph_tool network.
# If the deg. seq. has an uneven number of half-edges, a half-edge is added.
# Multi/Self edges are removed; degree distr. must be chosen so that the impact is
# negligible.
def construction_model(node_dist):
    
    node_dist_calc = node_dist
    # Checks if number of half-edges are uneven, adds 1 edge if necesarry
    if sum(node_dist) % 2 == 1:
        node_dist_calc[0] += 1 
        
    sum_node_dist_calc = sum(node_dist_calc)
    len_node_dist_calc = len(node_dist_calc)
    seq = np.zeros(int(sum_node_dist_calc))
    k = 0
    
    # Brings the dist. seq. into the half-edge form by
    # spreading the degree of a node over a number of indices of
    # the correspoding degree e.g. [1,2,3] -> [0,1,1,2,2,2]
    for i in range(len_node_dist_calc):
        for j in range(k, k + node_dist_calc[i]):
            seq[j] = i
            k += 1
            
    # Connects the half-edges randomly to each other by shuffling the
    # half-edge array
    np.random.shuffle(seq)
    links = np.reshape(seq, (int(len(seq)/2), 2))
    network = gt.Graph(directed=False)
    network.add_edge_list(links)
    
    gt.stats.remove_parallel_edges(network)
    gt.stats.remove_self_loops(network)
    
    return network


# In this version edges are prepared for linking with other modules. Maintains the deg. distr.
# Multi/Self edges are rewired 1000 times, to reduce the impact of removed egdes.
# Preferable to the version above.
def construction_model_c(node_dist, c):
    
    node_dist_calc = node_dist
        
    sum_node_dist_calc = sum(node_dist_calc)
    len_node_dist_calc = len(node_dist_calc)
    seq = []
    k = 0
    
    # Brings the dist. seq. into the half-edge form by
    # spreading the degree of a node over a number of indices of
    # the correspoding degree e.g. [1,2,3] -> [0,1,1,2,2,2]
    for i in range(len_node_dist_calc):
        for j in range(k, k + node_dist_calc[i]):
            seq += [i]
            k += 1
    
    c_node = np.random.choice(seq, size=c, replace=False)
    
    for i in range(c):
        seq.remove(c_node[i])
    
    # Checks if number of half-edges are uneven, removes 1 edge if necesarry
    if len(seq) % 2 == 1:
        del seq[0]
            
    # Connects the half-edges randomly to each other by shuffling the
    # half-edge array
    np.random.shuffle(seq)
    links = np.sort(np.reshape(seq, (int(len(seq)/2), 2)))
    links_unique = np.unique(links, axis=0).tolist()
    links = links.tolist()
    
    # Seperates the unique and the multi/self links.
    k = 0
    while k != len(links_unique):
        if links_unique[k][0]!=links_unique[k][1]:
            links.remove(links_unique[k])
            k+=1
        else:
            links_unique.remove(links_unique[k])
        
    # Flattens the multi/self link list.
    links_flat = []
    for sublist in links:
        for i in sublist:
            links_flat.append(i)    
        
    # Rewires the multi/self links n times and adds
    # them to the unique links, if not already in there.
    k = 0
    n = 1000
    while len(links_flat)>1 and k<n:
        r1 = random.choice(links_flat)
        r2 = random.choice(links_flat)
        if r1!=r2 and sorted([r1, r2]) not in links_unique:
            links_unique.append([r1, r2])
            links_flat.remove(r1)
            links_flat.remove(r2)
        k += 1
    
    network = gt.Graph(directed=False)
    network.add_edge_list(links_unique)
    
    gt.stats.remove_parallel_edges(network)
    gt.stats.remove_self_loops(network)
    
    return network, c_node


# Various versions of monodisperse modular networks as introduced in [...]
# Non-hierarchical, connected modular networks are created out of a power law degree
# distribution with given number and size of modules, degree distribution parameters
# kmin/kmax/g and number of inter modular connections.
# Utilizes the construction model, inverse transform sampling and returns networks
# in the graph_tool format.


# Creates intermod. links without changing the degree distr., using the prepared
# construction model. Standard inverse transform sampling.
def mmn(modul_number, modul_size, kmin, kmax, g, connections):
    
    check_unique = 0 # Checks if inter mod. con. are unique
    check_con = 0 # Checks if network is connected
        
    while check_unique != modul_number*connections/2 or check_con != 1:
    
        inter_nodes = np.zeros((modul_number, connections))
        network = gt.Graph(directed=False)
        # Constructs disconnected modules and combines them in a network
        # in the graph tool format.
        for i in range(modul_number):
            its = inverse_transform_sampling_correct(power_dist, g, kmin, kmax, modul_size)
            module_network, inter_nodes[i] = construction_model_c(its, connections)
            inter_nodes[i] += i*modul_size # Assigns the nodes to the corresponding module.
            network = gt.generation.graph_union(network, module_network)

        inter_nodes = np.transpose(inter_nodes)

        for row in inter_nodes:
            np.random.shuffle(row)

        inter_links = inter_nodes.ravel().reshape((int(modul_number*connections/2), 2))
        check_unique = len(np.unique(inter_links, axis=0))
        network.add_edge_list(inter_links)

        _, check_con = gt.topology.label_components(network)
        check_con = len(check_con)
        
    return network


# Creates intermod. links without changing the degree distr., using the prepared
# construction model. Offset inverse transform sampling.
def mmn_off(modul_number, modul_size, kmin, kmax, g, connections, offset):
    
    check_unique = 0 # Checks if inter mod. con. are unique
    check_con = 0 # Checks if network is connected
    
    while check_unique != modul_number*connections/2 or check_con != 1:
        
        inter_nodes = np.zeros((modul_number, connections))
        network = gt.Graph(directed=False)
        # Draws a degree distr. from a power law and creates a 
        for i in range(modul_number):
            its = inverse_transform_sampling_off(power_dist, g, kmin, kmax, modul_size, offset)
            module_network, inter_nodes[i] = construction_model_c(its, connections)
            inter_nodes[i] += i*modul_size
            network = gt.generation.graph_union(network, module_network)
        
        inter_nodes = np.transpose(inter_nodes)
        
        for row in inter_nodes:
            np.random.shuffle(row)

        inter_links = inter_nodes.ravel().reshape((int(modul_number*connections/2), 2))
        check_unique = len(np.unique(inter_links, axis=0))
        network.add_edge_list(inter_links)

        _, check_con = gt.topology.label_components(network)
        check_con = len(check_con)
        
    return network


# Generation of hierarchical modular networks as introduced in [...]
# Returns the networks in form of an edge list.


def hmn1(a, p, s, m0):
    """
    Creates a hierarchical-modular network with a 
    bottom-top approach.
    More information in 'Griffiths phases and the
    stretching of criticality' (see HMN1).
        
    Args:
        a, p: Determine linking probability
        s: Number of hierarchical levels
        m0: Size of building blocks
        
    Returns:
        List of edges.
    """
    N = 2*m0**s
    links = np.zeros((2*int(a/m0*N*sum([1/2**x for x in range(1, s+1)])), 2), dtype=np.int32)

    links_i = 0
    for si in range(1, s+1):
        
        # 2 Blocks at the same level get linked with
        # a shrinking probability with rising level.
        m0_si = m0**si
        for n in range(0, N+1-2*m0_si, 2*m0_si):
            
            # Connection ensures that at the end of each
            # linking process, at least one link between
            # blocks at the same level exists.
            m0_si_n = m0_si + n
            connection = 0
            while connection == 0:
                
                for link1 in range(n, m0_si_n):
                    for link2 in range(m0_si_n, 2*m0_si+n): 
                        if np.random.rand() < a*p**si:
                            links[links_i] = [link1, link2]
                            links_i += 1
                            connection = 1
                        
    # Creates the links in the blocks and removes empty
    # placeholders in links.
    blocks = np.arange(N).reshape((int(N/m0), m0))
    links = np.trim_zeros(links.ravel(), 'b').reshape((links_i, 2))

    return np.concatenate((blocks, links))


def hmn2(a, s, m0):
    """
    Creates a hierarchical-modular network with a 
    bottom-top approach.
    More information in 'Griffiths phases and the
    stretching of criticality' .
        
    Args:
        a: Number of links
        s: Number of hierarchical levels
        m0: Size of building blocks
        
    Returns:
        List of edges.
    """
    N = 2*m0**s
    links = np.zeros((int(a/m0*N*sum([1/2**x for x in range(1, s+1)])), 2), dtype=np.int32)
    links_i = 0
    p = 0
    
    # At each hierarchy level a number of a links are established,
    # repeating the process if links are repeated.
    for si in range(1, s+1):
        
        si_a = si*a
        m0_si = m0**si
        for n in range(0, N+1-2*m0_si, 2*m0_si):
            
            if a == 1:
                i = np.random.randint(0 + n, m0_si + n)
                j = np.random.randint(m0_si + n, 2*m0_si + n)
                links[p] = np.array([i, j])
                p += 1
            
            else:
                while len(np.unique(links[links_i:a + links_i], axis=0)) != a:
                    for m in range(a):
                        i = np.random.randint(0 + n, m0_si + n)
                        j = np.random.randint(m0_si + n, 2*m0_si + n)
                        links[links_i:a + links_i][m] = np.array([i, j])
                links_i += a
            
    blocks = np.arange(N).reshape((int(N/m0), m0))
    return np.concatenate((blocks, links))

