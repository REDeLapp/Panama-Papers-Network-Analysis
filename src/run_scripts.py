import networkx as nx
import load_and_graph2 as lg
import measuring2 as msr
import numpy as np
from collections import defaultdict
import pdb

def k_mean_cluster(G, filename = 'k_means'):
    '''
    GOAL: the function find the k_means with respective node labels
    and saves a Gephi network plot.
    ----------------------------------------------
    INPUT: instantiated networkx graph, G
    OUPUT:
    '''
    import numpy.linalg as la
    import scipy.cluster.vq as vq
    A = nx.adjacency_matrix(G)
    D = np.diag(np.ravel(np.sum(A,axis=1)))
    L = D - A
    # compute eigenvalues/eigenvectors using eig
    # 'l' is the evals and 'U' is the evects
    l, U = la.eigh(np.cov(L))
    f = U[:,1]
    labels = np.ravel(np.sign(f))
    k = 3
    means, lables = vq.kmeans2(U[:,1:k],k)
    lg.GeneralGraph(f, filename)
    return means, lables


def create_hc(ego, t = 1.15):
    """Creates hierarchical cluster of graph G from distance matrix
    ----------------------------------------------
    INPUT: G, instaciated networkx graph
    OUPUT: returns list of partition values split on hierarchical cluster"""
    # No other function uses these libraries, so i'm putting them within this function.
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    labels = ego.nodes()
    # Find the shortest path/edge between nodes
    path_length = dict(nx.all_pairs_shortest_path_length(ego)) # did this change?
    dist_matrix = np.zeros((len(ego), len(ego))) #distance matrix
    # d = [dist_matrix[u][v] for v, d in p.items() for u, p in path_length ]
    for u, p in path_length.items():
        for v, d in p.items():
            dist_matrix[u][v] = d
    # i = 0
    # for u,p in path_length.items():
    #     j = 0
    #     for v,d in p.items():
    #         dist_matrix[i][j] = d
    #         dist_matrix[j][i] = d
    #         if i==j:
    #             dist_matrix[i][j]=0
    #         j+=1
    #     i+=1

    # Create hierarchical cluster
    Y = distance.squareform(dist_matrix)
    Z = hierarchy.complete(Y)  # Creates HC using farthest point linkage
    membership = list(hierarchy.fcluster(Z, t=1.15))

    # Create collection of lists for blockmodel
    partition = defaultdict(list)
    for n, p in zip(list(range(len(ego))), membership):
        partition[p].append(labels[n])
    return list(partition.values())

def create_hc2(G,t=1.15):
    """Creates hierarchical cluster of graph G from distance matrix
    This return of this function is an argument to create a blockmodel with
    nx.quotient_graph...because nx.blockmodel is not supported by networkx v.2.0
    ----------------------------------------------
    INPUT:
    G, instaciated networkx graph
    t, is the threshold for partition selection, which is arbitrarity set to t=1.15 by default.
    OUPUT:
    returns list of partition values split on hierarchical cluster"""
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((len(G),len(G)))

    for u,p in path_length.items():
        for v,d in p.items():
            dist_matrix[u][v]=d

    # Create hierarchical cluster
    Y = distance.squareform(dist_matrix)

    # Creates HC using farthest point linkage
    Z = hierarchy.complete(Y)

    # This partition selection
    membership = list(hierarchy.fcluster(Z,t=t))

    # Create collection of lists for the blockmodel
    partition = defaultdict(list)
    for n,p in zip(list(range(len(G))), membership):
        partition[p].append(n)
    return list(partition.values())

def hiclus_blockmodel(G):
    """Draw a blockmodel diagram of a clustering"""
    # Extract largest connected component into graph H
    H = nx.connected_component_subgraphs(G)#[0]
    # Create parititions with hierarchical clustering
    cluster = create_hc(H)
    # cluster = create_hc(G)
    # Build blockmodel graph
    BM = nx.quotient_graph(H, cluster, create_using=nx.MultiGraph(), relabel = True)
    lg.GeneralGraph(GM,'HC_cluster_BM')
    # Draw original graph
    # pos=nx.spring_layout(H,iterations=100)
    # fig=plt.figure(1,figsize=(6,10))
    # ax=fig.add_subplot(211)
    # nx.draw(H,pos,with_labels=False,node_size=10)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # # Draw block model with weighted edges and nodes sized by
    # # number of internal nodes
    # node_size=[BM.node[x]['nnodes']*10 for x in BM.nodes()]
    # edge_width=[(2*d['weight']) for (u,v,d) in BM.edges(data=True)]
    # # Set positions to mean of positions of internal nodes from original graph
    # posBM={}
    # for n in BM:
    # xy=numpy.array([pos[u] for u in BM.node[n]['graph']])
    # posBM[n]=xy.mean(axis=0)
    # ax=fig.add_subplot(212)
    # nx.draw(BM,posBM,node_size=node_size,width=edge_width,with_labels=False)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.axis('off')

def plot_hist_size_partition(partition):
    unique_size = len(unique(list(partition.values())))
    plt.hist(partition.values(), bins = unique_size)
    plt.xlabel('The Number of Node In Partition')
    plt.ylabel('Frequency of Partition Size')
    plt.title('Distribution of Partition Size in Nodes')

if __name__ == '__main__':

    G, all_nodes = lg.load_clean_data() # Import and clean
    ego = lg.build_subgraph(G, all_nodes ) # Create subgroup
    # lg.GeneralGraph(ego, filename = "replace_nan_unknown_no_regex") #Generages image in Gephi
    # f = filter_nodess_by_degree(ego, all_nodes, k=3)
    # Q = msr.my_louvian_modularity(ego)
    print(" Finished loading network graphs G and ego! ")
    '''
    What does create_HC look like?
    '''
    # Extract largest connected component into graph H
    H = next(nx.connected_component_subgraphs(ego))
    # labele all integer nodes
    H = nx.convert_node_labels_to_integers(H)
    # Create parititions with hierarchical clustering
    cluster1 = create_hc(H)

    # # Build blockmodel graph
    # BM = nx.quotient_graph(H, partitions, relabel=True)
    #
    # # Draw original graph
    # pos = nx.spring_layout(H, iterations=100)
    # plt.subplot(211)
    # nx.draw(H, pos, with_labels=False, node_size=10)
    #
    # # Draw block model with weighted edges and nodes sized by number of internal nodes
    # node_size = [BM.nodes[x]['nnodes'] * 10 for x in BM.nodes()]
    # edge_width = [(2 * d['weight']) for (u, v, d) in BM.edges(data=True)]
    # # Set positions to mean of positions of internal nodes from original graph
    # posBM = {}
    # for n in BM:
    #     xy = numpy.array([pos[u] for u in BM.nodes[n]['graph']])
    #     posBM[n] = xy.mean(axis=0)
    # plt.subplot(212)
    # nx.draw(BM, posBM, node_size=node_size, width=edge_width, with_labels=False)
    # plt.axis('off')
    # plt.show()
    #
    # '''
    # Playing with block model
    # '''
    #
    # cluster = create_hc(ego)
    # BM = nx.quotient_graph(ego, clusters, create_using=nx.MultiGraph(), relabel = True)
    # lg.GeneralGraph(BM, 'hc_cluster_ego')
    # df_centralities = msr.comparing_centralities(ego)
    # md_centralities = pandas_df_to_markdown_table(df_centralities)
    #
    # dgre_connectivity_dict, mean_degrees = msr.edge_analysis(ego, attr = 'intermediary')
    # msr.plot_hist_avg_degrees(ego, attr = 'intermediary')
