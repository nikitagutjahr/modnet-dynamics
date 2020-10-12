#!/usr/bin/env python

"""
Implementations for monodisperse and hierarchical modular networks.
Functions before those are helper functions.
"""
import random
import numpy as np
import scipy as sp
import graph_tool as gt
from graph_tool.all import *


def inverse_transform_sampling(g, kmin, kmax, size, offset):
    """Inverse transform sampling of a power-law distribution.

    Used in the configuration model.
    Generates a degree list from a power-law distribution.
    The offset subtracts degrees up to a minimum of kmin.

    Args:
        g (float): Power-law exponent
        kmin (int): Min degree
        kmax (int): Max degree
        size (int): Number of nodes
        offset (int): Degree subtraction

    Returns:
        list: Degrees of indexed nodes
    """
    # Power-law distribution
    c = sp.special.zeta(g)
    def dist(k): return 1/(np.asarray(k)**g*c)

    # Calculates the probability to find nodes with degree up to kmax and
    # the corresponding cumulative distribution.
    dist_n = dist(range(kmin, kmax+1))
    cumsum = np.cumsum(dist_n)
    rand = np.random.rand(size) * cumsum[-1]
    node_dist = np.array([kmin]*size)
    
    # Checks in which region of the cum. distr. the random values lie
    # and assigns the corresponding node degree.
    for i in range(size):
        j = offset
        while (cumsum[j] - rand[i]) < 0:
            node_dist[i] += 1
            j += 1
    
    return node_dist.astype(int)


def configuration_model(g, kmin, kmax, size, c, offset):
    """Generates a network via the configuration model.

    Edges are prepared for linking with other modules so that the
    degree distr. is maintained. Multi/self edges are rewired once
    within the modules, then removed.

    Args:
        g (float): Power-law exponent
        kmin (int): Min degree
        kmax (int): Max degree
        size (int): Number of nodes
        c (int): Number of connections to other modules
        offset (int): Degree subtraction

    Returns:
        gt.Graph: Generated network
        list: Edges prepared for intermod. linking
    """
    # Makes sure that networks have an even number of half-edges, since
    # the number of half-edges prepared for linking can be uneven.
    its = inverse_transform_sampling(g, kmin, kmax, size, offset)
    sum_its = sum(its)
    while sum_its % 2 != c % 2:
        its = inverse_transform_sampling(g, kmin, kmax, size, offset)
        sum_its = sum(its)

    # Brings the dist. seq. into the half-edge form,
    # e.g. [1,2,3] -> [0,1,1,2,2,2]
    seq = []
    k = 0
    for i in range(size):
        for j in range(k, k + its[i]):
            seq += [i]
            k += 1

    # Edges that link to other modules
    c_node = np.random.choice(seq, size=c, replace=False)
    for i in range(c):
        seq.remove(c_node[i])
    
    # Connects the half-edges randomly among each other
    np.random.shuffle(seq)
    links = np.sort(np.reshape(seq, (int(len(seq)/2), 2)))
    links = links.tolist()
    
    # Separates the unique and the multi/self links.
    # The list links is left with multi and self links.
    links_unique = []
    k = 0
    while k < len(links):
        if links[k][0] != links[k][1] and links[k] not in links_unique:
            links_unique.append(links.pop(k))
        else:
            k += 1
        
    # Rewires the multi/self links
    for multi in links:
        r1 = random.randint(0, len(links_unique)-1)
        r2 = random.randint(0, 1)
        r3 = random.randint(0, 1)
        links_unique[r1][r2], multi[r3] = multi[r3], links_unique[r1][r2]
    
    network = gt.Graph(directed=False)
    network.add_edge_list(links_unique)
    network.add_edge_list(links)

    gt.stats.remove_parallel_edges(network)
    gt.stats.remove_self_loops(network)
    
    return network, c_node


def mmn(modul_number, modul_size, kmin, kmax, g, c, offset):
    """Generates a monodisperse modular network.

    MMNs are non-hierarchical, connected modular networks.
    They consist of sparsely interconnected power-law modules.
    For details see README.

    Args:
        modul_number (int): Number of modules
        modul_size (int): Number of nodes in modules
        kmin (int): Min degree
        kmax (int): Max degree
        g (float): Power-law exponent
        c (int): Number of intermod. connections
        offset (int): Degree subtraction

    Returns:
        gt.Graph: MMN
    """
    check_unique = 0  # Checks if inter mod. con. are unique
    check_con = 0  # Checks if network is connected
    while check_unique != modul_number*c/2 or check_con != 1:
        inter_nodes = np.zeros((modul_number, c))
        network = gt.Graph(directed=False)
        # Constructs disconnected modules and combines them in a network
        # in the graph tool format.
        for i in range(modul_number):
            module_network, inter_nodes[i] = configuration_model(
                                           g, kmin, kmax,
                                           modul_size, c, offset)
            # Assigns the nodes to the corresponding module.
            inter_nodes[i] += i*modul_size
            network = gt.generation.graph_union(network, module_network)

        inter_nodes = np.transpose(inter_nodes)
        for row in inter_nodes:
            np.random.shuffle(row)

        inter_links = inter_nodes.ravel().reshape((int(modul_number*c/2), 2))
        check_unique = len(np.unique(inter_links, axis=0))
        network.add_edge_list(inter_links)

        _, check_con = gt.topology.label_components(network)
        check_con = len(check_con)
        
    return network


def hmn1(a, p, s, m0):
    """Generates a hierarchical-modular network type 1

    More information in 'Griffiths phases and the
    stretching of criticality' (see HMN1).
        
    Args:
        a (int): Determines linking probability
        p (float): Determines linking probability
        s (int): Number of hierarchical levels
        m0 (int): Size of building blocks
        
    Returns:
        List: Edge list
    """
    n = 2*m0**s
    links = np.zeros((2*int(a/m0*n*sum([1/2**x for x in range(1, s+1)])), 2), dtype=np.int32)

    links_i = 0
    for si in range(1, s+1):
        # 2 Blocks at the same level get linked with
        # a shrinking probability with rising level.
        m0_si = m0**si
        for n in range(0, n+1-2*m0_si, 2*m0_si):
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
    blocks = np.arange(n).reshape((int(n/m0), m0))
    links = np.trim_zeros(links.ravel(), 'b').reshape((links_i, 2))

    return np.concatenate((blocks, links))


def hmn2(a, s, m0):
    """Generates a hierarchical-modular network type 2

    More information in 'Griffiths phases and the
    stretching of criticality' (see HMN2).
        
    Args:
        a (int): Number of links
        s (int): Number of hierarchical levels
        m0 (int): Size of building blocks
        
    Returns:
        List: Edge list
    """
    n = 2*m0**s
    links = np.zeros((int(a/m0*n*sum([1/2**x for x in range(1, s+1)])), 2), dtype=np.int32)
    links_i = 0
    p = 0
    
    # At each hierarchy level a number of a links are established,
    # repeating the process if links are repeated.
    for si in range(1, s+1):
        m0_si = m0**si
        for n in range(0, n+1-2*m0_si, 2*m0_si):
            
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
            
    blocks = np.arange(n).reshape((int(n/m0), m0))
    return np.concatenate((blocks, links))
