# -*- coding: utf-8 -*-
"""
A NetworkX based implementation of Yen's algorithm for computing K-shortest paths.   
Yen's algorithm computes single-source K-shortest loopless paths for a 
graph with non-negative edge cost. For more details, see: 
http://en.m.wikipedia.org/wiki/Yen%27s_algorithm
"""
from collections import deque
from heapq import heappush, heappop
from itertools import count
import networkx as nx
from networkx.utils import generate_unique_node
import copy
from datetime import datetime
import math
def get_sec(time_str):
    try:
        h, m, s = time_str.split(':')
    except:
        error = repr(time_str)+' '+str(type(time_str))
        raise RuntimeError(error)
    return float(h) * 3600 + float(m) * 60 + float(s)



def get_string_date(time_float):
    hours = int(time_float // 3600)
    time_float -= hours*3600
    minutes = int(time_float) // 60
    time_float -= minutes*60
    second = int(time_float)
    return "{:02d}:{:02d}:{:02d}".format(hours,minutes,second)

# This function is inspired from the original networkx code base
def single_source_dijkstra(G, source, target=None, cutoff=None, weight='weight',start_time='00:00:00'):
    """Compute shortest paths and lengths in a weighted graph G.

    Uses Dijkstra's algorithm for shortest paths.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the path from the source to that node.


    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length,path=nx.single_source_dijkstra(G,0)
    >>> print(length[4])
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    ---------
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path()
    single_source_dijkstra_path_length()
    """
    if source == target:
        return ({source: 0}, {source: [(source,None,None,None)]})
    if cutoff is not None:
        cutoff = get_sec(cutoff)
    


    paths = {source: [(source,None,None,None)]}  # dictionary of paths
    return _dijkstra(G, source, paths=paths, cutoff=cutoff,
                     target=target,time_start=start_time)

def get_weight(edge,current_time,pred_tupple):
    '''
    Used to compute the cost of of an edge prunes edges with departure time < current_time
    and  do not consider an edges if its a connection and there is less than 1mn to catch the next train
    '''
    thresh = current_time
    if (edge['type'] == 'walk'):
        dep_time = thresh
        waiting = 0
    else:
        if (pred_tupple[2] is not None) and (pred_tupple[2] != edge['trip_id']) and pred_tupple[3] != 'walk' :
            thresh += 60
        dep_time = get_sec(edge['S_departure_time'])
        if (dep_time < thresh):
            return None
        waiting = dep_time - current_time
        
    return edge['scheduled_trip_time']+ waiting
def _dijkstra(G,source,time_start='00:00:00',pred=None, paths=None, cutoff=None,
              target=None):
    """Implementation of Dijkstra's algorithm

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    get_weight: function
        Function for getting edge weight

    pred: list, optional(default=None)
        List of predecessors of a node

    paths: dict, optional (default=None)
        Path from the source to a target node.

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the path from the source to that node.

    pred,distance : dictionaries
       Returns two dictionaries representing a list of predecessors
       of a node and the distance to each node.

    distance : dictionary
       Dictionary of shortest lengths keyed by target.
    """
    G_succ = G.succ if G.is_directed() else G.adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    #nb_walks = 
    time_start = get_sec(time_start)
    seen = {source: time_start}
    prev = dict.fromkeys(list(G.nodes), (None, None, None,None)) # Pred node,pred_edge_id,pred_trip_id,pred_type
    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (time_start, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        current_time = dist[v]
        for u, e_group in G_succ[v].items():
            for id_,e in e_group.items():
                tmp = (v,id_,e['trip_id'],e['type'])
                cost = get_weight(e, current_time, prev[v]) 
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                if u in dist:
                    if vu_dist < dist[u]:
                        raise ValueError('Contradictory paths found:',
                                         'negative weights?')
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    prev[u] = tmp
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = copy.deepcopy(paths[v]) + [tmp]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)

    if paths is not None:
        return (dist, paths)
    if pred is not None:
        return (dist, pred)
    return dist,None

def index_graph(G,id_1,id_2,id_3):
    return G[id_1][id_2][id_3]
def get_path_edges_ids(dst,path):
    edges =[]
    for i in range(1,len(path)):
        p_i = path[i]
        if i < len(path)-1:
            edges.append((p_i[0],path[i+1][0],p_i[1]))
        else:
            edges.append((p_i[0],dst,p_i[1]))

    return edges

def remove_worst_edge(network, edges):
    '''
    removes the worst edge from the network and returns it
    returns None if the network is already good enough (total uncertainty >= threshold)
    '''
    min_arg = 0
    min_val = 1000
    lambdas = [network[x[0]][x[1]][x[2]]['lambda_arrival_delay'] for x in edges]
    for i in range(len(lambdas)):
        if lambdas[i] < min_val and index_graph(network,*edges[i])['type'] != 'walk':
            min_val =  lambdas[i]
            min_arg = i
    edge = edges[min_arg]
    network.remove_edge(*edge)

    
    

def path_certainty(G,edges,dst,function_proba,start_time):
    """
    Take a path and determine the rate of missing each transport change according to the predictive model
    we built earlier. It returns also the combining ratio for the whole path.
    """
    cum_proba = 0.
    indiv_probas = [1]
    for i in range(1,len(edges)): 
        edge_1 = index_graph(G,*edges[i-1])
        if i == 1 and edge_1['type'] == 'walk':
            edge_1['S_departure_time'] = start_time
            edge_1['S_arrival_time'] = get_string_date(get_sec(start_time) + edge_1['scheduled_trip_time'])
        edge_2 = index_graph(G,*edges[i])
        if edge_2['type'] == 'walk':
            tmp = edge_1['S_arrival_time']
            edge_2['S_departure_time'] = tmp
            edge_2['S_arrival_time'] = get_string_date(get_sec(tmp) + int(edge_2['scheduled_trip_time']))
            edge_2['lambda_arrival_delay'] = edge_1['lambda_arrival_delay']
            indiv_probas.append(1)
        else:
            if edge_1['trip_id'] == edge_2['trip_id']:
                indiv_probas.append(1)
            else:
                arr_time = edge_1['S_arrival_time']
                dep_time = edge_2['S_departure_time']
                proba = function_proba(edge_1['lambda_arrival_delay'],get_sec(dep_time)-get_sec(arr_time))
                #proba = 1
                indiv_probas.append(proba)
                cum_proba += math.log(proba)
    return indiv_probas,math.exp(cum_proba)

def safest_path(G, source, dst,function_proba, start_time='00:00:00', n_iters=10, threshold=0.8, total_uncertainty=None, path=None, cutoff=None):
    '''
    tries to compute new shortest path by iteratively removing the most risky edge of the network
    and recomputing a shortest path from the reduced network. 
    If the maximum number of iteration is reached,we return the last computed shortest path 
    '''
    # do a deepcopy of the network we need
    my_net = G.copy()
    
    if path is not None and total_uncertainty < threshold :
        remove_worst_edge(my_net, path)
    
    all_paths = []
    i = 0
    while(i < n_iters):
        distance,sp = single_source_dijkstra(my_net, source, target=dst,start_time=start_time, cutoff=cutoff)
        if dst not in distance:
            return None,None,None,None
        distance = distance[dst]
        sp = sp[dst]
        edges = get_path_edges_ids(dst,sp)
        dep_time = [G[x[0]][x[1]][x[2]]['S_departure_time'] for x in edges]
        lambdas = [G[x[0]][x[1]][x[2]]['lambda_arrival_delay'] for x in edges]
        types = [G[x[0]][x[1]][x[2]]['type'] for x in edges]
 
        uncertainty_array,global_uncertainty = path_certainty(G,edges,dst,function_proba,start_time)
        # stop iterating if the path is safe enough
        if global_uncertainty >= threshold:
            return edges,global_uncertainty,uncertainty_array,get_string_date(distance)
        remove_worst_edge(my_net, edges)
        i += 1
    return edges,global_uncertainty,uncertainty_array,get_string_date(distance)

def get_isochrone(G, source, function_proba, start_time='00:00:00', cutoff='00:30:00', n_iters=10, threshold=0.8):
    '''
    Find all reachable stations from a source node given a starting time, a maximum arrival time and an uncertainy threshold 
    '''
    
    all_paths = single_source_dijkstra(G, source,start_time=start_time, cutoff=cutoff)
    all_distance,all_paths = single_source_dijkstra(G, source,start_time=start_time, cutoff=cutoff)
        
    for dst, path in all_paths.items():
        distance = get_string_date(all_distance[dst])
        edges = get_path_edges_ids(dst, path)
        uncertainty_array, global_uncertainty = path_certainty(G, edges, dst, function_proba, start_time)
        
        if global_uncertainty < threshold:
            edges, global_uncertainty, uncertainty_array,distance = safest_path(G, source, dst, function_proba, start_time=start_time, n_iters=n_iters, threshold=threshold, total_uncertainty=global_uncertainty, path=edges, cutoff=cutoff)
            if edges is None or global_uncertainty < threshold:
                continue
        
        yield dst, edges, global_uncertainty, distance
      
    
    