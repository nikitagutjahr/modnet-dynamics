#!/usr/bin/env python
#cython: language_level=3

"""
Various implementations of the SIS and the CP spreading process.
For detailed descriptions of the algorithms see the articles listed in
the README.

TODO:
    * Optimize contact_process_qs
    * Only fully active initial states are allowed. That can be
      generalized to allow all initial states
"""

import numpy as np
cimport numpy as np
import random
cimport cython
import copy

@cython.wraparound(False)
@cython.boundscheck(False)


def sis(float rate, int t, np.ndarray[long, ndim=1, negative_indices=False,
        mode='c'] start, list neighbor, float kmax, int n):
    """
    Executes the SIS on a network.

    For detailed descriptions see articles listed in README.
    
    Args:
        rate (float): Spreading rate
        t (int): Simulation length
        start (np.array): Initial state (must be fully active)
        neighbor (list[np.array]): Network as list of neighbors
        kmax (int): Maximal degree of network
        n (int): Random seed
    
    Returns:
        np.array: Activity density for each time step.
    """
    np.random.seed(n)
    random.seed(n)
    cdef int rand_size = 10000000
    cdef float s
    cdef int network_size = len(neighbor)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                    mode='c'] density = np.zeros(t+1, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                    mode='c'] active = np.copy(start)
    cdef int Ne = sum([len(neighbor[i]) for i in range(network_size)])
    cdef set inactive = set()
    cdef int Ni = len(start)  # Assumes fully active initial state
    density[0] = Ni/network_size
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef int node_deg
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                    mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    
    while time_length < t and Ni > 0:
        # Update time step
        dyn_R = Ni + rate*Ne
        time_length += 1/dyn_R

        # Random number preparation. Faster then generating them
        # individually at each step.
        if rn_index >= rand_size-10000:
            rand = np.random.rand(rand_size)
            rn_index = random.randint(0, rand_size-1)

        # Process selection: Either a node deactivates,
        # or it activates a neighbor node.
        s = Ni/dyn_R
        if rand[rn_index] < s:
            rn_index += 1
            node_index = int(rand[rn_index]*Ni)
            rn_index += 1
            node_deg = len(neighbor[active[node_index] - 1])
            Ne -= node_deg
            Ni -= 1
            inactive.add(active[node_index])
            active[node_index] = active[Ni]  # Note different usages of Ni

        else:
            rn_index += 1
            node_deg = 0
            while rand[rn_index] >= node_deg/kmax:  # Rejection method
                rn_index += 1
                node_index = int(rand[rn_index]*Ni)  # Rnd node to spread from
                rn_index += 1
                node_deg = len(neighbor[active[node_index] - 1])

            rn_index += 1
            # Rnd node to spread to
            rand_node_index = int(rand[rn_index]*node_deg)
            rn_index += 1
            rand_node = neighbor[active[node_index] - 1][rand_node_index]

            if rand_node in inactive:
                inactive.remove(rand_node)
                active[Ni] = rand_node
                node_deg = len(neighbor[rand_node-1])
                Ne += node_deg
                Ni += 1

            # Record density at time step
            while time_length >= step:
                density[step] = Ni/network_size
                step += 1

    return density[:step]


def sis_qs(float rate, int t, np.ndarray[long, ndim=1, negative_indices=False,
           mode='c'] start, list neighbor, float kmax, int M,
           int n, int rand_size=10000000):
    """ Executes the QS SIS on a network.

    For detailed descriptions see articles listed in README.

    Args:
        rate (float): Spreading rate
        t (int): Simulation length
        start (np.array): Initial state (must be fully active)
        neighbor (list[np.array]): Network as list of neighbors
        kmax (int): Maximal degree
        M (int): Size of QS memory
        n (int): Random seed

    Returns:
        np.array: Activity density for each time step.
    """
    np.random.seed(n)
    random.seed(n)
    cdef float s
    cdef int network_size = len(neighbor)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                    mode='c'] density = np.zeros(t, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                    mode='c'] active = np.copy(start)
    cdef int Ne = sum([len(neighbor[i]) for i in range(network_size)])
    cdef set inactive = set(range(1, network_size)) - set(start)
    cdef double Ni = len(start)  # Assumes fully active initial state
    density[0] = Ni/network_size
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef int node_deg
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                    mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    cdef int check = 0
    
    cdef np.ndarray[long, ndim=2, negative_indices=False,
                    mode='c'] backup_active = np.zeros((M, network_size),
                    dtype=np.int64)
    backup_active[0] = active
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                    mode='c'] backup_Ne = np.zeros(M, dtype=np.int64)
    backup_Ne[0] = Ne
    
    while time_length < t:
        # Update time step
        dyn_R = Ni + rate*Ne
        time_length += 1/dyn_R

        # Random number preparation. Faster then generating them
        # individually at each step.
        if rn_index >= rand_size-1000:
            rand = np.random.rand(rand_size)
            rn_index = random.randint(0, rand_size-1)
            check += 1
            
        # Process selection: Either a node deactivates,
        # or it activates a neighbor node.
        s = Ni/dyn_R
        if rand[rn_index] < s:
            rn_index += 1
            node_index = int(rand[rn_index]*Ni)
            rn_index += 1
            node_deg = len(neighbor[active[node_index] - 1])
            Ne -= node_deg
            Ni -= 1
            inactive.add(active[node_index])
            active[node_index] = active[int(Ni)]
            active[int(Ni)] = 0

        else:
            rn_index += 1
            node_deg = 0

            while rand[rn_index] >= node_deg/kmax:  # Rejection method
                rn_index += 1
                node_index = int(rand[rn_index]*Ni)  # Rnd node to spread from
                rn_index += 1
                node_deg = len(neighbor[active[node_index] - 1])

            rn_index += 1
            # Rnd node to spread to
            rand_node_index = int(rand[rn_index]*node_deg)
            rn_index += 1
            rand_node = neighbor[active[node_index] - 1][rand_node_index]

            if rand_node in inactive:
                inactive.remove(rand_node)
                active[int(Ni)] = rand_node
                node_deg = len(neighbor[rand_node - 1])
                Ne += node_deg
                Ni += 1

            while time_length >= step:
                density[step] = Ni/network_size

                # Backup of the network state
                if step < M:
                    backup_active[step] = copy.copy(active)
                    backup_Ne[step] = copy.copy(Ne)

                else:
                    if random.random() <= 0.01:
                        rand_M = random.randint(0, M - 1)
                        backup_active[rand_M] = copy.copy(active)
                        backup_Ne[rand_M] = copy.copy(Ne)
                step += 1

        # QS implementation keeps the process alive,
        # by returning the inactive network to a previous state.
        if Ni == 0:
            m = min(step, M)
            rand_m = random.randint(0, m-1)
            active = copy.copy(backup_active[:m][rand_m])
            Ne = copy.copy(backup_Ne[:m][rand_m])
            Ni = len(np.trim_zeros(active, 'b'))
            inactive = set(range(network_size)) - set(active)

    return density


def contact_process(float rate, int t, np.ndarray[long, ndim=1,
                    negative_indices=False, mode='c'] start, list neighbor,
                    int n, int rand_size=10000000):
    """
    Executes the CP on a network.

    For detailed descriptions see articles listed in README.

    Args:
        rate (float): Spreading rate
        t (int): Simulation length
        start (np.array): Initial state (must be fully active)
        neighbor (list[np.array]): Network as list of neighbors
        n (int): Random seed

    Returns:
        np.array: Activity density for each time step.
    """
    np.random.seed(n)
    random.seed(n)
    cdef float s = rate/(rate + 1)
    cdef int network_size = len(neighbor)
    #avalanche_size = np.zeros(t, dtype=np.int32)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                    mode='c'] density = np.zeros(t, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                    mode='c'] active = np.copy(start)
    cdef set inactive = set(range(1, network_size)) - set(start)
    cdef double Ni = len(start) # Assumes fully active initial state
    #avalanche_size[0] = Ni
    density[0] = Ni/network_size
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef double node_deg
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                    mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    cdef int check = 0
    
    # Active nodes are saved in an array from its left side upwards.
    # Inactive nodes are additionally saved in a set,
    # for quicker access.
    while time_length < t:
        
        # Random number preparation. Faster then generating them
        # individually at each step.
        if rn_index >= rand_size-2:
            rand = np.random.rand(rand_size)
            rn_index = rn_index = random.randint(0, rand_size-1)
            check += 1
            
        # Node selection: Either the chosen node deactivates,
        # or it activates a neighbor node.
        node_index = int(rand[rn_index]*Ni) # Rnd node to spread from
        rn_index += 1
        if rand[rn_index] < s:
            rn_index += 1
            node_deg = len(neighbor[active[node_index]-1])
            # Rnd node to spread to
            rand_node_index = int(rand[rn_index]*node_deg)
            rn_index += 1
            rand_node = neighbor[active[node_index]-1][rand_node_index]
        
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[int(Ni)] = rand_node
                Ni += 1

        else:
            rn_index += 1
            Ni -= 1
            inactive.add(active[node_index])
            active[node_index] = active[int(Ni)]
            active[int(Ni)] = 0

        if time_length >= step:
            #avalanche_size[step] = Ni
            density[step] = Ni/network_size
            step += 1
        
        # If process dies out.
        if Ni == 0:
            break
        
        time_length = time_length + 1/Ni

    return density[:step]


def contact_process_qs(rate, t, start, m, neighbor):
    """ Executes the QS CP on a network.

    For detailed descriptions see articles listed in README.
    Needs to be optimized.

    Args:
        rate (float): Spreading rate
        t (int): Simulation length
        start (np.array): Initial state (must be fully active)
        m (int): Size of QS memory
        neighbor (list[np.array]): Network as list of neighbors

    Returns:
        np.array: Activity density for each time step.
    """
    s = rate/(rate + 1)
    network_size = len(neighbor)
    avalanche_size = np.zeros(t, dtype=np.int32)
    density = np.zeros(t, dtype=np.float32)
    active = copy.copy(start)
    inactive = set(range(network_size)) - set(start)
    ni = len(start) # Assumes fully active initial state
    avalanche_size[0] = ni
    density[0] = avalanche_size[0]/network_size
    time_length = 0
    step = 1
    backup_active = np.zeros((m, network_size), dtype=np.int32)
    backup_active[0] = active

    # Active nodes are saved in an array from its left side upwards.
    # Inactive nodes are additionally saved in a set, for quicker access.
    while time_length < t:
        
        # Node selection: Either the chosen node deactivates,
        # or it activates a neighbor node.
        node_index = random.choice(range(ni))
        
        if random.random() <= s:
            rand_node = random.choice(neighbor[active[node_index]-1])
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[ni] = rand_node
                ni += 1

        else:
            ni -= 1
            inactive.add(active[node_index])
            active[node_index] = active[ni]
            active[ni] = 0

        # QS implementation keeps the process alive,
        # by returning the empty network to a previous state.
        if ni == 0:
            m = min(step, m)
            active = copy.copy(backup_active[:m][random.randint(0, m-1)])
            ni = len(np.trim_zeros(active, 'b'))
            inactive = set(range(network_size)) - set(active)
            
        # System state is saved after an time increment of 1.
        if time_length >= step:
            avalanche_size[step] = ni
            density[step] = avalanche_size[step]/network_size
            
            if step < m:
                backup_active[step] = copy.copy(active)
                
            else:
                if random.random() <= 0.001:
                    backup_active[random.randint(0, m - 1)] = copy.copy(active)
            
            step += 1
        
        time_length = time_length + 1 / ni

    return density