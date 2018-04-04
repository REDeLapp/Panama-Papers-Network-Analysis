import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as cm
from operator import itemgetter

def comparing_centralities(G):
#     ## Comparing the Centralities
#     dgr = nx.degree_centrality(ego)
    col = nx.closeness_centrality(ego)
    har = nx.harmonic_centrality(ego)
    bet = nx.betweenness_centrality(ego)
    dgr = nx.degree_centrality(ego)
    pgr = nx.pagerank(ego)
    eig = nx.eigenvector_centrality(ego)
#
    centralities = pd.concat(
    [pd.Series(c) for c in (eig, pgr, har, bet, col)], axis = 1)

    centralities.columns("Eigenvector", "PageRank",
                        "Harmonic Closeness", "Closeness",
                        "Degree", "Betweenness")
    centralities["Harmonic Closeness"] /= centralities.shape[0]

    # Calculated the correlations for each pair of centralities
    c_df = centralities.corr()
    ll_triangle = np.tri(c_df.shape[0], k= -1)
    c_df *= ll_triangle
    c_series = c_df.stack().sort_values()
    c_series.tail()

def max_sort_centrality(mydict, num):
    '''
    INPUT:
    - mydict, give a dictionary of centrality
    - num, the number to
    OUPUT: returns the node_id index on the names of the
    '''
    sred = sorted(mydict.items(), key=lambda value:float(value[1]))
    sorted(mydict, key=mydict.get).lemit(num)

def plot_harmonic_closeness_eigen(G):
    '''
    "A network with positive correlation attributes is called
    assortative; in an asssortative network, nodes tend to connect to nodes
    with similar attribute values. This tendency is called assortative mixing.
    A dissortative newtork is the opposite." ~ book
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

    ## Comparing the Centralities
    # dgr = nx.degree_centrality(ego)
    col = nx.closeness_centrality(ego)
    har = nx.harmonic_centrality(ego)

    bet = nx.betweenness_centrality(ego)
    dgr = nx.degree_centrality(ego)
    pgr = nx.pagerank(ego)
    eig = nx.eigenvector_centrality(ego)

    centralities = pd.concat(
    [pd.Series(c) for c in (eig, pgr, har, clo, dgr, bet)], axis = 1)

    centralities.columns("Eigenvector", "PageRank",
                        "Harmonic Closeness", "Closeness",
                        "Degree", "Betweenness")
    centralities["Harmonic Closeness"] /= centralities.shape[0]

    # Calculated the correlations for each pair of centralities
    c_df = centralities.corr()
    ll_triangle = np.tri(c_df.shape[0], k= -1)
    c_df *= ll_triangle
    c_series = c_df.stack().sort_values()
    c_series.tail()


    nx.attribute_mixing_matrix(G, "country", mapping = {"SUA": 0, "JOR": 1})
