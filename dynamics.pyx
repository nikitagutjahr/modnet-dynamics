#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Various implementations of the SIS and the CP spreading process.
# For CP details see [...].
# For SIS details see [...].
# For the quasistationary method details see [...].

import numpy as np
cimport numpy as np
import random
cimport cython
import copy

@cython.wraparound(False)
@cython.boundscheck(False)

# The code got increasingly complex for opimization purposes.
# contact_process_simpe is an unoptimized version, included for easier
# understanding of the process.

def contact_process_simple(network, rate, t, start):
    """
    Executes the CP on a network, by going
    through the active sites and either deactivating
    them, or spreading the activity to one neighbor.
    
    Args:
        network: Adj. matrix of a network
        rate: Spreading rate
        t: Process length
        start: Initial activity state
    
    Returns:
        List containing avalanche size, density
        for each time step.
    """
    s = rate/(rate + 1)
    len_network = network.shape[0]
    avalanche_size = np.zeros(t)
    density = np.zeros(t)
    activity = set(start)
    avalanche_size[0] = len(activity)
    density[0] = avalanche_size[0]/len_network
    time_length = 0
    step = 1
    neighbor = np.array(list(map(np.nonzero, network)))[:, 1]

    while time_length < t:
        
        # This takes a lot of performance due to conversion
        # of data type.
        j = random.choice(tuple(activity)) 
        
        # Node selection: Either the choosen node deactivates,
        # or it activates a neighbor node
        if random.random() <= s:
            activity.add(random.choice(neighbor[j]))

        else:
            activity.remove(j)
        
        len_activity = len(activity)
        if time_length >= step:
            avalanche_size[step] = len_activity
            density[step] = avalanche_size[step]/len_network
            step += 1
        
        if activity == set():
            break # If process dies out.
        time_length = time_length + 1/len_activity

    return avalanche_size[:step], density[:step]


# Depending on network type, the neighbor dimension has to changed (lattice=2, other=1),
# also neighbor dtype (lattice=int, other=np.int)
def contact_process(float rate, int t, np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] start, np.ndarray[np.int, ndim=1, negative_indices=False,
                mode='c'] neighbor, int n, int rand_size=10000000):
    """
    Executes the CP on a network, by going
    through the active sites and either deactivating
    them, or spreading the activity to one neighbor.
    
    Args:
        rate: Spreading rate
        t: Simulation length
        start: Initial state (must be fully active)
        neighbor: Network in the neighbor array format
        n: Random seed
    
    Returns:
        Array with activity density for each time step.
    """
    np.random.seed(n)
    random.seed(n)
    cdef float s = rate/(rate + 1)
    cdef int len_network = len(neighbor)
    #avalanche_size = np.zeros(t, dtype=np.int32)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                mode='c'] density = np.zeros(t, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] active = np.copy(start)
    cdef set inactive = set(range(1, len_network)) - set(start)
    cdef double len_active = len(start) # Assumes fully active initial state
    #avalanche_size[0] = len_active
    density[0] = len_active/len_network
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef double len_neighbor
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                   mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    cdef int check = 0
    
    # Active nodes are saved in an array from its left side upwards.
    # Unactive nodes are additionally saved in a set, for quicker access.
    while time_length < t:
        
        # Pregenerating random numbers is faster.
        if rn_index >= rand_size-2:
            rand = np.random.rand(rand_size)
            rn_index = rn_index = random.randint(0, rand_size-1)
            check += 1
            
        # Node selection: Either the choosen node deactivates,
        # or it activates a neighbor node.
        
        node_index = int(rand[rn_index]*len_active) # Rnd node to spread from
        rn_index += 1
        if rand[rn_index] < s:
            rn_index += 1
            len_neighbor = len(neighbor[active[node_index]-1])
            rand_node_index = int(rand[rn_index]*len_neighbor) # Rnd node to spread to
            rn_index += 1
            rand_node = neighbor[active[node_index]-1][rand_node_index]
        
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[int(len_active)] = rand_node
                # len_active updated at the end due to different usages
                # of the variable. Either as indice, or as
                # the actual length.
                len_active += 1

        else:
            rn_index += 1
            len_active -= 1 # Same as above
            inactive.add(active[node_index])
            active[node_index] = active[int(len_active)]
            active[int(len_active)] = 0

        if time_length >= step:
            #avalanche_size[step] = len_active
            density[step] = len_active/len_network
            step += 1
        
        # If process dies out.
        if len_active == 0:
            break #
        
        time_length = time_length + 1/len_active

    return density[:step]


# Not optimized with cython. Has to be adjusted like the sis_qs.
def contact_process_qs(rate, t, start, M, neighbor):
    """
    Executes the CP on a network, by going
    through the active sites and either deactivating
    them, or spreading the activity to one neighbor.
    Utilized the quasistationary method.
    
    Args:
        rate: Spreading rate
        t: Simulation length
        start: Initial state (must be fully active)
        M: Size of QS memory
        neighbor: Network in the neighbor array format
    
    Returns:
        Array with activity density for each time step.
    """
    s = rate/(rate + 1)
    len_network = len(neighbor)
    avalanche_size = np.zeros(t, dtype=np.int32)
    density = np.zeros(t, dtype=np.float32)
    active = copy.copy(start)
    inactive = set(range(len_network)) - set(start)
    len_active = len(start) # Assumes fully active initial state
    avalanche_size[0] = len_active
    density[0] = avalanche_size[0]/len_network
    time_length = 0
    step = 1
    backup_active = np.zeros((M, len_network), dtype=np.int32)
    backup_active[0] = active

    # Active nodes are saved in an array from its left side upwards.
    # Unactive nodes are additionally saved in a set, for quicker access.
    while time_length < t:
        
        # Node selection: Either the choosen node deactivates,
        # or it activates a neighbor node.
        node_index = random.choice(range(len_active))
        
        if random.random() <= s:
            rand_node = random.choice(neighbor[active[node_index]-1])
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[len_active] = rand_node
                # len_active updated at the end due to different usages
                # of the variable. Either as indice, or as
                # the actual length.
                len_active += 1

        else:
            len_active -= 1 # Same as above
            inactive.add(active[node_index])
            active[node_index] = active[len_active]
            active[len_active] = 0

        # QS implementation keeps the process alive,
        # by returning the empty network to a previous state.
        if len_active == 0:
            m = min(step, M)
            active = copy.copy(backup_active[:m][random.randint(0, m-1)])
            len_active = len(np.trim_zeros(active, 'b'))
            inactive = set(range(len_network)) - set(active)
            
        # System state is saved after an time increment of 1.
        if time_length >= step:
            avalanche_size[step] = len_active
            density[step] = avalanche_size[step]/len_network
            
            if step < M:
                backup_active[step] = copy.copy(active)
                
            else:
                if random.random() <= 0.001:
                    backup_active[random.randint(0, M-1)] = copy.copy(active)
            
            step += 1
        
        time_length = time_length + 1/len_active

    return density


# Depending on network type, the neighbor dimension has to changed (lattice=2, other=1),
# also neighbor dtype (lattice=int, other=np.int)
def sis(float rate, int t, np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] start, np.ndarray[np.int, ndim=1, negative_indices=False,
                mode='c'] neighbor, float kmax, int n, int rand_size=10000000):
    """
    Executes the SIS on a network, by going
    through the active sites and either deactivating
    them, or spreading the activity to one neighbor.
    
   Args:
        rate: Spreading rate
        t: Simulation length
        start: Initial state (must be fully active)
        neighbor: Network in the neighbor array format
        kmax: Maximal degree
        n: Random seed
    
    Returns:
        Array with activity density for each time step.
    """
    np.random.seed(n)
    random.seed(n)
    cdef float s
    cdef int len_network = len(neighbor)
    #avalanche_size = np.zeros(t, dtype=np.int32)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                mode='c'] density = np.zeros(t, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] active = np.copy(start)
    cdef int Ne = sum([len(neighbor[i]) for i in range(len_network)])
    cdef set inactive = set(range(1, len_network)) - set(start)
    cdef double len_active = len(start) # Assumes fully active initial state
    #avalanche_size[0] = len_active
    density[0] = len_active/len_network
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef int len_neighbor
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                   mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    cdef int check = 0
    
    # Active nodes are saved in an array from its left side upwards.
    # Unactive nodes are additionally saved in a set, for quicker access.
    while time_length < t:
        
        # Random number preparation
        if rn_index >= rand_size-100:
            rand = np.random.rand(rand_size)
            rn_index = random.randint(0, rand_size-1)
            check += 1
            
        # Process selection: Either a node deactivates,
        # or it activates a neighbor node.
        
        s = len_active/(len_active+rate*Ne)
        if rand[rn_index] >= s:
            rn_index += 1
            len_neighbor = 0
            
            while rand[rn_index] >= len_neighbor/(2*kmax): # Rejection method
                rn_index += 1
                node_index = int(rand[rn_index]*len_active) # Rnd node to spread from
                rn_index += 1
                len_neighbor = len(neighbor[active[node_index]-1])
                
            rn_index += 1
            rand_node_index = int(rand[rn_index]*len_neighbor) # Rnd node to spread to
            rn_index += 1
            rand_node = neighbor[active[node_index]-1][rand_node_index]
        
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[int(len_active)] = rand_node
                len_neighbor = len(neighbor[rand_node-1])
                Ne += len_neighbor
                # len_active updated at the end due to different usages
                # of the variable. Either as indice, or as
                # the actual length.
                len_active += 1

        else:
            rn_index += 1
            node_index = int(rand[rn_index]*len_active)
            rn_index += 1
            len_neighbor = len(neighbor[active[node_index]-1])
            Ne -= len_neighbor
            len_active -= 1 # Same as above
            inactive.add(active[node_index])
            active[node_index] = active[int(len_active)]
            active[int(len_active)] = 0
        
        if time_length >= step:
            #avalanche_size[step] = len_active
            density[step] = len_active/len_network
            step += 1
            
        if len_active == 0:
            break # If process dies out.
        
        #time_length = time_length + 1/len_active
        time_length = time_length + 1/(len_active+rate*Ne)

    #return avalanche_size[:step], density[:step]
    return density[:step]


def sis_qs(float rate, int t, np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] start, np.ndarray[np.int, ndim=1, negative_indices=False,
                mode='c'] neighbor, float kmax, int M, int n, int rand_size=10000000):
    """
    Executes the SIS on a network, by going
    through the active sites and either deactivating
    them, or spreading the activity to one neighbor.
    Ulitilized the QS method.
    
   Args:
        rate: Spreading rate
        t: Simulation length
        start: Initial state (must be fully active)
        neighbor: Network in the neighbor array format
        kmax: Maximal degree
        m: QS memory
        n: Random seed
    
    Returns:
        Array with activity density for each time step.
    """
    
    np.random.seed(n)
    random.seed(n)
    cdef float s
    cdef int len_network = len(neighbor)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                mode='c'] density = np.zeros(t, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] active = np.copy(start)
    cdef int Ne = sum([len(neighbor[i]) for i in range(len_network)])
    cdef set inactive = set(range(1, len_network)) - set(start)
    cdef double len_active = len(start) # Assumes fully active initial state
    density[0] = len_active/len_network
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef int len_neighbor
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                   mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    cdef int check = 0
    
    cdef np.ndarray[long, ndim=2, negative_indices=False,
                   mode='c'] backup_active = np.zeros((M, len_network), dtype=np.int64)
    backup_active[0] = active
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                   mode='c'] backup_Ne = np.zeros(M, dtype=np.int64)
    backup_Ne[0] = Ne
    
    # Active nodes are saved in an array from its left side upwards.
    # Unactive nodes are additionally saved in a set, for quicker access.
    while time_length < t:
        
        # Random number preparation
        if rn_index >= rand_size-100:
            rand = np.random.rand(rand_size)
            rn_index = random.randint(0, rand_size-1)
            check += 1
            
        # Process selection: Either a node deactivates,
        # or it activates a neighbor node.
        
        s = len_active/(len_active+rate*Ne)
        if rand[rn_index] >= s:
            rn_index += 1
            len_neighbor = 0
            
            while rand[rn_index] >= len_neighbor/(2*kmax): # Rejection method
                rn_index += 1
                node_index = int(rand[rn_index]*len_active) # Rnd node to spread from
                rn_index += 1
                len_neighbor = len(neighbor[active[node_index]-1])
                
            rn_index += 1
            rand_node_index = int(rand[rn_index]*len_neighbor) # Rnd node to spread to
            rn_index += 1
            rand_node = neighbor[active[node_index]-1][rand_node_index]
        
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[int(len_active)] = rand_node
                len_neighbor = len(neighbor[rand_node-1])
                Ne += len_neighbor
                # len_active updated at the end due to different usages
                # of the variable. Either as indice, or as
                # the actual length.
                len_active += 1

        else:
            rn_index += 1
            node_index = int(rand[rn_index]*len_active)
            rn_index += 1
            len_neighbor = len(neighbor[active[node_index]-1])
            Ne -= len_neighbor
            len_active -= 1 # Same as above
            inactive.add(active[node_index])
            active[node_index] = active[int(len_active)]
            active[int(len_active)] = 0
                    
        # QS implementation keeps the process alive,
        # by returning the empty network to a previous state.
        if len_active == 0:
            m = min(step, M)
            rand_m = random.randint(0, m-1)
            active = copy.copy(backup_active[:m][rand_m])
            Ne = copy.copy(backup_Ne[:m][rand_m])
            len_active = len(np.trim_zeros(active, 'b'))
            inactive = set(range(len_network)) - set(active)
            
        # System state is saved after an time increment of 1.
        if time_length >= step:
            density[step] = len_active/len_network
            
            if step < M:
                backup_active[step] = copy.copy(active)
                backup_Ne[step] = copy.copy(Ne)

            else:
                if random.random() <= 0.01:
                    rand_M = random.randint(0, M-1)
                    backup_active[rand_M] = copy.copy(active)
                    backup_Ne[rand_M] = copy.copy(Ne)
            
            step += 1

        time_length = time_length + 1/(len_active+rate*Ne)
 
    return density


# Depending on network type, the neighbor dimension has to changed (lattice=2, other=1),
# also neighbor dtype (lattice=int, other=np.int)
def sis(float rate, int t, np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] start, np.ndarray[np.int, ndim=1, negative_indices=False,
                mode='c'] neighbor, float kmax, int n, int rand_size=10000000):
    """
    Executes the SIS on a network, by going
    through the active sites and either deactivating
    them, or spreading the activity to one neighbor.
    
   Args:
        rate: Spreading rate
        t: Simulation length
        start: Initial state (must be fully active)
        neighbor: Network in the neighbor array format
        kmax: Maximal degree
        n: Random seed
    
    Returns:
        Array with activity density for each time step.
    """
    np.random.seed(n)
    random.seed(n)
    cdef float s
    cdef int len_network = len(neighbor)
    #avalanche_size = np.zeros(t, dtype=np.int32)
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                mode='c'] density = np.zeros(t, dtype=np.float)
    cdef np.ndarray[long, ndim=1, negative_indices=False,
                mode='c'] active = np.copy(start)
    cdef int Ne = sum([len(neighbor[i]) for i in range(len_network)])
    cdef set inactive = set(range(1, len_network)) - set(start)
    cdef double len_active = len(start) # Assumes fully active initial state
    #avalanche_size[0] = len_active
    density[0] = len_active/len_network
    cdef double time_length = 0
    cdef int step = 1
    cdef int node_index
    cdef int rand_node
    cdef int rand_node_index
    cdef int len_neighbor
    cdef np.ndarray[double, ndim=1, negative_indices=False,
                   mode='c'] rand = np.random.rand(rand_size)
    cdef int rn_index = random.randint(0, rand_size-1)
    cdef int check = 0
    
    # Active nodes are saved in an array from its left side upwards.
    # Unactive nodes are additionally saved in a set, for quicker access.
    while time_length < t:
        
        # Random number preparation
        if rn_index >= rand_size-100:
            rand = np.random.rand(rand_size)
            rn_index = random.randint(0, rand_size-1)
            check += 1
            
        # Process selection: Either a node deactivates,
        # or it activates a neighbor node.
        
        s = len_active/(len_active+rate*Ne)
        if rand[rn_index] >= s:
            rn_index += 1
            len_neighbor = 0
            
            while rand[rn_index] >= len_neighbor/(2*kmax): # Rejection method
                rn_index += 1
                node_index = int(rand[rn_index]*len_active) # Rnd node to spread from
                rn_index += 1
                len_neighbor = len(neighbor[active[node_index]-1])
                
            rn_index += 1
            rand_node_index = int(rand[rn_index]*len_neighbor) # Rnd node to spread to
            rn_index += 1
            rand_node = neighbor[active[node_index]-1][rand_node_index]
        
            if rand_node in inactive:
                inactive.remove(rand_node)
                active[int(len_active)] = rand_node
                len_neighbor = len(neighbor[rand_node-1])
                Ne += len_neighbor
                # len_active updated at the end due to different usages
                # of the variable. Either as indice, or as
                # the actual length.
                len_active += 1

        else:
            rn_index += 1
            node_index = int(rand[rn_index]*len_active)
            rn_index += 1
            len_neighbor = len(neighbor[active[node_index]-1])
            Ne -= len_neighbor
            len_active -= 1 # Same as above
            inactive.add(active[node_index])
            active[node_index] = active[int(len_active)]
            active[int(len_active)] = 0
        
        if time_length >= step:
            #avalanche_size[step] = len_active
            density[step] = len_active/len_network
            step += 1
            
        if len_active == 0:
            break # If process dies out.
        
        #time_length = time_length + 1/len_active
        time_length = time_length + 1/(len_active+rate*Ne)

    #return avalanche_size[:step], density[:step]
    return density[:step]

