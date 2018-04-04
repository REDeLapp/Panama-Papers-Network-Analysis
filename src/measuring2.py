import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as cm
from operator import itemgetter

def comparing_centralities(G):
    '''
    This function campares centralities
    -------------------------------------------
    INPUT:
    - G, the instanciated networkx graph
    OUPUT: Returns a data frame of all pair wise correlation between the centralities
    '''
    col = nx.closeness_centrality(ego)
    har = nx.harmonic_centrality(ego)
    bet = nx.betweenness_centrality(ego)
    dgr = nx.degree_centrality(ego)
    pgr = nx.pagerank(ego)
    eig = nx.eigenvector_centrality(ego, max_iter = 500)
    # Converts into pandas series
    centralities = pd.concat([pd.Series(c) for c in (eig, pgr, har, col, dgr, bet)], axis = 1)
    # Concatentates all vectors into a pd.DataFrame
    centralities.columns = ["Eigenvector", "PageRank", "Harmonic Closeness", "Closeness", "Degree", "Betweenness"]
    # Harmonic Closeness is the only NetworkX centrality that is not returned normalized
    centralities["Harmonic Closeness"] /= centralities.shape[0] # Normalize the Harmonic Closeness

    # Calculated the correlations for each pair of centralities
    c_df = centralities.corr()
    ll_triangle = np.tri(c_df.shape[0], k= -1)
    c_df *= ll_triangle
    c_series = c_df.stack().sort_values()
    c_series.tail()

def max_sort_centrality(mydict, num):
    '''
    -------------------------------------------
    INPUT:
    - mydict, give a dictionary of centrality
    - num, the number to
    OUPUT: returns the node_id index on the names of the
    '''
    sred = sorted(mydict.items(), key=lambda value:float(value[1]))
    sorted(mydict, key=mydict.get).lemit(num)

def plot_harmonic_closeness_eigen(G):
    '''
    GAOL
    "A network with positive correlation attributes is called
    assortative; in an asssortative network, nodes tend to connect to nodes
    with similar attribute values. This tendency is called assortative mixing.
    A dissortative newtork is the opposite." ~ book
    -------------------------------------------
    INPUT: G, networkx graph
    OUPUT:
    scatter plot of the eignevector centralities versus the closeness centralities
    '''
    eig = nx.eigenvector_centrality(G, max_iter = 1000)
    col = nx.closeness_centrality(G)
    ax = plt.gca()
    ax.scatter(col,eig , c='black', alpha=0.5, edgecolors='none')
    # ax.abline(intercept=0, slope=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    pass

def plot_avg_nieghbor_degree_node_degree(G):
    '''
    Estimate Network Uniformity Through Assortativity
    -------------------------------------------
    INPUT: Networkx graph, G
    OUPUT: A scatter plot of the node degree versus Avg Nieghbor degree'''
    ax = plt.gca()
    my_degree, their_degree = zip(*nx.average_degree_connectivity(G).items())
    ax.scatter(my_degree, their_degree , c='black', alpha=0.5, edgecolors='none')
    # ax.abline(intercept=0, slope=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    pass

def my_attribute_assortativity_coefficient(G, feature):
    '''

    INPUT:
    - 'G' is an instantiated of a networkX graph
    - 'feature' is a string of an attribute/feature you want to explore
    OUTPUT: the assortativity coefficient for a specific attribute
    '''
    return nx.attribute_assortativity_coefficient(G, feature)

def my_louvian_modularity(G):
    '''
    THe modularity, m, is the fraction of edges that fall within the given
    communities minus the expected fraction if edges were distributed at random,
    while conserving the node's degrees.
    ------------------
    interpretation:
    -----------------

    INPUT: instanciated networkx graph, G
    OUTPUT: the louvian modularity
    '''
    partition = community.best_partition(ego)
    return cm.modularity(partition, G)

# def jake_modularity(G):
    # return m
    pass
def modularity_based_communities(G):
    '''
    ----------------------------------
    INPUT: Networkx graph network
    OUPUT:
    -the communities,
    - their sizes,
    - which nodes belong to which community
    '''
    partition = cm.best_partition(G)

    part_as_series = pd.Series(partition)
    part_as_series.sort_values()
    pass

if __name__ == '__main__':


    nx.attribute_mixing_matrix(G, "country", mapping = {"SUA": 0, "JOR": 1})
