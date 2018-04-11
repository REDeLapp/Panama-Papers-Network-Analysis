import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as cm
from operator import itemgetter
from collections import Counter, defaultdict
import load_and_graph2 as lg

def all_my_centralities(G):
    '''
    GOAL: all the centralities
    ----------------------------------------------
    INPUT: G, a networkx graph.
    OUPUT: a list of Dictionary of nodes with degree centrality as the value
            for each centrality metrics.
    '''
    col = nx.closeness_centrality(G)
    har = nx.harmonic_centrality(G)
    bet = nx.betweenness_centrality(G)
    dgr = nx.degree_centrality(G)
    pgr = nx.pagerank(G)
    eig = nx.eigenvector_centrality(G, max_iter = 500)
    hits = nx.hits(G)

    har /= len(har) # Normalize the Harmonic Closeness

    return [col, har, bet, dgr, pgr, eig, hit[0]]
## Centrality related functions
def comparing_centralities(G):
    '''
    This function campares centralities
    -------------------------------------------
    INPUT:
    - G, the instantiatednetworkx graph
    OUPUT: Returns a data frame of all pair wise correlation
    between the centralities
    '''
    col = nx.closeness_centrality(G)
    har = nx.harmonic_centrality(G)
    bet = nx.betweenness_centrality(G)
    dgr = nx.degree_centrality(G)
    pgr = nx.pagerank(G)
    eig = nx.eigenvector_centrality(G, max_iter = 500)
    hits = nx.hits(G)

    # Converts into pandas series
    centralities = pd.concat([pd.Series(c) for c in (eig, pgr, har, col, dgr, bet, hits[0])], axis = 1)
    # Concatentates all vectors into a pd.DataFrame
    centralities.columns = ["Eigenvector", "PageRank", "Harmonic Closeness", "Closeness", "Degree", "Betweenness", "Hubs"]
    # Harmonic Closeness is the only NetworkX centrality that is not returned normalized
    centralities["Harmonic Closeness"] /= centralities.shape[0] # Normalize the Harmonic Closeness

    # Calculated the correlations for each pair of centralities
    c_df = centralities.corr()
    ll_triangle = np.tri(c_df.shape[0], k= -1)
    c_df *= ll_triangle
    c_series = c_df.stack().sort_values()
    c_series.tail()
    # return centralities, c_series.tail()
def top_nodes_with_highest_degree(G, k):
    '''
    GOAL: Find the top ten nodes with the highest degrees
    ----------------------------------------------
    INPUT:
    G, networkx graph
    k, the number to top nodes you want to see

    OUPUT: a sorted list of tuple of the names of the nodes with the repective degrees
    '''
    deg = dict(nx.degree(G))
    top10 = sorted([(n, G.node[n]["type"], v) for n, v in deg.items()],
               key=lambda x: x[2], reverse=True)[:k]

    print("\n".join(["{} ({}): {}".format(*t) for t in top10]))
    return top10

def max_sort_centrality(mydict, num):
    '''
    GOAL:
    -------------------------------------------
    INPUT:
    - mydict, give a dictionary of centralies
    - num, the limit of the top maximum centralities
    OUPUT: returns the node_id index on the names of the
    '''

    # top10 = sorted([(n, G.node[n]["type"], v) for n, v in deg.items()],
    #            key=lambda x: x[2], reverse=True)[:10]
    #
    # print("\n".join(["{} ({}): {}".format(*t) for t in top10]))

    sred = sorted(mydict.items(), key=lambda value:float(value[1]))
    return sorted(mydict, key=mydict.get).limit(num)

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
    # ax.abline(np.log(1):np.log(3), 1000/np.log(1):1000/np.log(3):)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('Node Degree')
    plt.ylabel('Average Neighbor Degree')
    plt.title('A scatter plot of the node degree versus Avg Nieghbor degree')
    plt.show()

def my_attribute_assortativity_coefficient(G, feature):
    '''
    GOAL:
    ----------------------------------------------
    INPUT:
    - 'G' is an instantiated of a networkX graph
    - 'feature' is a string of an attribute/feature you want to explore
    OUTPUT: the assortativity coefficient for a specific attribute
    '''
    return nx.attribute_assortativity_coefficient(G, feature)

def my_louvian_modularity(G):
    '''
    GAOL: THe modularity, Q, is the fraction of edges that fall within the given
    communities minus the expected fraction if edges were distributed at random,
    while conserving the node's degrees.
    ------------------
    interpretation:
    -----------------

    INPUT: instantiated networkx graph, G
    OUTPUT: the louvian modularity, a float
    '''
    partition = cm.best_partition(G)
    return cm.modularity(partition, G)

def modularity_based_communities(G):
    '''
    GAOL: to partition the graph by its louvian modularity and return 'k' communites
    with the hightest modularity.
    ----------------------------------------------
    INPUT: Networkx graph network
    OUPUT:
    - partition_as_series, the communities and which nodes belong to which community
    - community_sizes, is a pandas.core.series.Series of the Names and number of communities
    '''
    ## Partitions
    # 1)dictionary with node labels as keys and integer communite identifiers as value
    # 2) calc the mdoualrity of the partition with respect to the orignial network.   sd
    partition = cm.best_partition(G) # Louvian Modularity algorithm
    # Convert network partition into a pandas series
    partition_as_series = pd.Series(partition) # unsorted
    partition_as_series_sorted = partition_as_series.sort_values(ascending=False) #
    community_size = partition_as_series.value_counts() #size of communite
    return partition_as_series_sorted, community_size

def plot_hist_len_connected_components(G):
    x = [len(c) for c in net.connected_component_subgraphs(G)]
    plot.hist()

## Tools
def filter_nodes_by_degree(G, all_nodes, k=2):
    '''
    GOAL: this function removes all nodes of a graph G of
    degrees less than or equal to k
    --------------------------------------
    INPUT:
    - G, instantiated networkx graphs
    - 'all_nodes', is the concatenation of all node lists into one dataframe
    - k, the number of degree on which you want to filter. It is default to two for this model.
    OUPUT: returns a networkx graph
    '''
    # filter nodes with degree less than k
    nodes = all_nodes.reindex(G)
    nodes = nodes[~nodes.index.duplicated()]
    f = nx.Graph()
    deg = dict(nx.degree(G))
    fedges = filter(lambda x: deg[x[0]] > k and deg[x[1]] > k, G.edges())
    f.add_edges_from(fedges)

    # # reattach attribute
    # nx.set_node_attributes(f, nodes["country_codes"], "cc")
    # nx.set_node_attributes(f, nodes["type"], "ty")
    # nx.set_node_attributes(f, nodes["name"], "nm")
    # # get rid of null and turn the list into a dictionary
    # valid_names = nodes[nodes["name"].notnull()]["name"].to_dict()
    # # f = nx.relabel_nodes(f, nodes[nodes.name.notnull()].name
    # #                      & nodes.name.isnull()].address)
    # nx.relabel_nodes(f, valid_names)
    f = reattach_attributes_of_interest(f, all_nodes)
    return f

def edge_analysis(G, attr = 'intermediary'):
    '''
    Find the average number of degree a node with a certain attribute
    ----------------------------------------------
    INPUT:
    - G, instantiated newtorkx graph
    - attr, is a string of the attribute you want to explore
    OUPUT:
    - 'degrees_connectivity_dict', is a dictionary of the
    - 'mean_degrees', the average number of degrees the nodes
        of a certain attribute has
    '''
    # node_inter = filter(lambda n, d: d['type'] == 'intermediary', G.nodes(data=True))
    # intermediaries = pd.read_csv('/Users/rdelapp/Galvanize/DSI_g61/capstone/panama_papers/data/csv_panama_papers_2018-02-14/panama_papers_nodes_intermediary.csv', index_col = "node_id")
    # idx_nodes = all_nodes[all_nodes.type == 'intermediary'].index
    # foo = intermediaries.name
    nodes_of_interest = node_attr(G, attr = attr)
    degrees_connectivity_dict = nx.average_degree_connectivity(G, nodes = nodes_of_interest)

    # nodes_of_interest = [x for x,y in ego.nodes(data=True) if y['ty']== 'intermediary' ]
    # nx.average_degree_connectivity(ego, nodes = nodes_of_interest)
    mean_degrees = [v  for k,v in G.degree(nodes_of_interest)]

    return degrees_connectivity_dict, mean_degrees

def node_attr(G, attr = 'ty', name = 'intermediary'):
    '''
    GOAL: return only the node of a certain attribute, attr
    ----------------------------------------------
    INPUT:
    -G, instantiatedNewtorkx graph
    - attr, is the attritbute on which you want to select nodes: ty, cc, nm
    - name, is the name of the attribute: for type(ty) you want all 'intermediary'
    OUPUT:
    returns all nodes that have a specific attribute
    '''
    return [x for x,y in G.nodes(data=True) if y[attr] == name]

def pandas_df_to_markdown_table(df):
    '''
    GOAL: this function converts a Pandas DataFrame into a markdown table.
    ----------------------------------------------
    INPUT: df, a pandas data frame
    OUPUT: a markdown table
    '''
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))

def plot_hist_avg_degrees(G, attr):
    '''
    Visualizes the distribution of the average degrees for an
    attribute of interest.
    Can either be passed 'G' and 'attr' or just the 'mean_degrees'.
    ----------------------------------------------
    INPUT:
    - G, a instantiatednetworkx graph
    - attr, the attribute of interest
    - mean_degrees, the average number of degrees the nodes
        of a certain attribute has.
    OUPUT:
    - returns histogram of mean_degrees
    '''
    # tmp is a list of degrees_connectivity_dict, mean_degrees
    tmp = edge_analysis(G, attr = 'intermediary')
    plt.hist(tmp[1], bins = 50)
    plt.xlabel('The Mean Degree Per Node For {}'.format(attr))
    plt.ylabel('Frequency of Mean Degree')
    plt.title('Distribution of Mean Degrees For {} Nodes'.format(attr))
    # else:
    #     plt.hist(mean_degeers, bins = 20)

def plot_degree_vs_frequency(G):
    '''
    GOAL:
    ----------------------------------------------
    INPUT:
    OUPUT:
    '''
    deg = dict(nx.degree(G))
    x,y = zip(*Counter(deg.values()).items())
    ax = plt.gca()
    plt.scatter(x,y)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Barabasi-Albert Network Check')
    ax.set_xscale('log')

def plot_degree_vs_clustering(G,ego):
    """
    The clustering coeff is a measure of hte prevalence of triangles in an egocentric netowrk
    The clustering coeff is a fraction of possible triangles that contain the ego node and the exist
    ----------------------------------------------
    """
    deg = nx.degree(G)
    cc = nx.clustering(nx.Graph(G),ego)
    ptl.scatter(x,y)
    plt.xlabel('Degrees')
    plt.ylabel('Clustering Coefficient (cc)')
    plt.title('Degrees Versus Clustering Coefficient')
    # else:
# nx.attribute_mixing_matrix(G, "country", mapping = {"SUA": 0, "JOR": 1})
def plot_hist_size_partition(partition):
    '''
    ----------------------------------------------
    '''
    unique_size = len(unique(list(partition.values())))
    plt.hist(partition.values(), bins = unique_size)
    plt.xlabel('The Number of Node In Partition')
    plt.ylabel('Frequency of Partition Size')
    plt.title('Distribution of Partition Size in Nodes')

def Community_break_down(ego):

    ## get list of nodes for each louvian_partition
    community_list = defaultdict(list)
    for k,v in dendo[0].items():
        community_list[v].append(k)

    ## generate separate subgraph for each community
    ego_list = []
    for n in community_list.keys(): # goes through index of communities based on modularity
        nx.subgraph(ego, community_list[n])
        # ego = reattach_attributes_of_interest(ego, all_nodes)
        ego_list.append(nx.subgraph(ego, community_list[n]))
    # ego_list = [nx.subgraph(ego, community_list[n]), for n in community_list.keys()]

    # get modularity values per subgraph
    ego_modularity = []
    for i in range(len(community_list)):
        Q = cm.modularity(community_list[i], ego_list[i])
        ego_modularity.append(Q)

    # return top ranking modularity communites

    # break down each community
    '''
    1. Country_community

    2. avg_degree of community

    3. community centrality

    4. nodes with top degrees
    '''
