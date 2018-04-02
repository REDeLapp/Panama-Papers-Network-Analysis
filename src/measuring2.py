import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_harmonic_closeness_eigen(G):
    pass

def plot_avg_nieghbor_degree_node_degree(G):
    pass

def my_attribute_assortativity_coefficient(G, feature):
    '''
    INPUT:
    - 'G' is an instantiated of a networkX graph
    - 'feature' is a string of an attribute/feature you want to explore
    '''
    return nx.attribute_assortativity_coefficient(G, feature)

if __name__ == '__main__':

## Comparing the Centralities
dgr = nx.degree_centrality(G)
col = nx.closeness_centrality(G)
har = nx.harmonic_centrality(G)
eig = nx.eigenvector_centrality(G)
bet = nx.betweenness_centrality(G)
pgr = nx.pagerank(G)

centralities = pd.concat(
[pd.Series(c) for c in (hits[1], eig, pgr, har, clo, hits[0], dgr, bet)], axis = 1)

centralities.columns("Authorities", "Eigenvector", "PageRank",
                    "Harmonic Closeness", "Closeness", "Hubs",
                    "Degree", "Betweenness")
centralities["Harmonic Closeness"] /= centralities.shape[0]

# Calculated the correlations for each pair of centralities
c_df = centralities.corr()
ll_triangle = np.tri(c_df.shape[0], k= -1)
c_df *= ll_triangle
c_series = c_df.stack().sort_values()
c_series.tail()

X = "Harmonic Closeness"
Y = "Eigenvector"
limits = pd.concat([centralities[[X, Y]].min(),
                    centralities[[X, Y]].max()], axis = 1).values
centralities.plot(kind = "scatter", x=X, y=Y, xlim = limits[0], ylim = limits[1],
                  s=75, logy=True, alpha = 0.6)

## Estimate Network Uniformity Through Assortativity
my_degree, their_degree = zip(*nx.average_degree_conneectivity(G).items())

nx.attribute_mixing_matrix(G, "country", mapping = {"SUA": 0, "JOR": 1})

def my_attribute_assortativity_coefficient(G, feature):
    '''
    INPUT:
    - istantiated _____, G
    - feature that is tobe explored, 'feature'
    '''
    return nx.attribute_assortativity_coefficient(G, feature)
