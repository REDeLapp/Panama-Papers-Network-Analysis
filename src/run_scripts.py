import networkx as nx
import load_and_graph2 as lg
import measuring2 as msr

# def filter_nodes_by_degree(G, all_nodes, k):
#     '''
#     GOAL: this function removes all nodes of a graph G of
#     degrees less than or equal to k
#     --------------------------------------
#     INPUT:
#     - G, instantiated networkx graphs
#     - 'all_nodes', is the concatenation of all node lists into one dataframe
#     - k, the number of degree on which you want to filter
#     OUPUT: returns a networkx graph
#     '''
#     # filter nodes with degree less than k
#     nodes = all_nodes.reindex(G)
#     nodes = nodes[~nodes.index.duplicated()]
#     f = nx.Graph()
#     fedges = filter(lambda x: G.degree()[x[0]] > k and G.degree()[x[1]] > k, G.edges())
#     f.add_edges_from(fedges)
#
#     # reattach attribute
#     nx.set_node_attributes(f, nodes["country_codes"], "cc")
#     nx.set_node_attributes(f, nodes["type"], "ty")
#     nx.set_node_attributes(f, nodes["name"], "nm")
#     # get rid of null and turn the list into a dictionary
#     valid_names = nodes[nodes["name"].notnull()]["name"].to_dict()
#     f = nx.relabel_nodes(f, nodes[nodes.name.notnull()].name)
#     # ego = nx.relabel_nodes(ego, nodes[nodes.address.notnull()
#     #                                 & nodes.name.isnull()].address)
#     nx.relabel_nodes(f, valid_names)
#     return f
#
# def edge_analysis(G, attr = 'intermediary'):
#     '''
#     Find the average number of degree a node with a certain attribute
#
#     INPUT:
#     - G, instantiated newtorkx graph
#     - attr, is a string of the attribute you want to explore
#     OUPUT:
#     - 'dgre_connectivity_dict', is a dictionary of the
#     - 'mean_degrees', the average number of degrees the nodes
#         of a certain attribute has
#     '''
#     # node_inter = filter(lambda n, d: d['type'] == 'intermediary', G.nodes(data=True))
#     # intermediaries = pd.read_csv('/Users/rdelapp/Galvanize/DSI_g61/capstone/panama_papers/data/csv_panama_papers_2018-02-14/panama_papers_nodes_intermediary.csv', index_col = "node_id")
#     # idx_nodes = all_nodes[all_nodes.type == 'intermediary'].index
#     # foo = intermediaries.name
#     nodes_of_interest = node_attr(G, attr = 'intermediary')
#     dgr_connectivity_dict = nx.average_degree_connectivity(G, nodes = nodes_of_interest)
#
#     # nodes_of_interest = [x for x,y in ego.nodes(data=True) if y['ty']== 'intermediary' ]
#     # nx.average_degree_connectivity(ego, nodes = nodes_of_interest)
#     mean_degrees = [v  for k,v in ego.degree(nodes_of_interest)]
#
#     return dgre_connectivity_dict, mean_degrees
#
# def node_attr(G, attr = 'intermediary'):
#     return [x for x,y in G.nodes(data=True) if y['ty'] == attr]
#
# def pandas_df_to_markdown_table(df):
#     '''
#     GOAL: this function converts a Pandas DataFrame into a markdown table.
#     INPUT: df, a pandas data frame
#     OUPUT: a markdown table
#     '''
#     from IPython.display import Markdown, display
#     fmt = ['---' for i in range(len(df.columns))]
#     df_fmt = pd.DataFrame([fmt], columns=df.columns)
#     df_formatted = pd.concat([df_fmt, df])
#     display(Markdown(df_formatted.to_csv(sep="|", index=False)))
#
# def plot_hist_avg_degrees(G, attr):
#     '''
#     Visualizes the distribution of the average degrees for an
#     attribute of interest.
#     Can either be passed 'G' and 'attr' or just the 'mean_degrees'.
#     INPUT:
#     - G, a instanciated networkx graph
#     - attr, the attribute of interest
#     - mean_degrees, the average number of degrees the nodes
#         of a certain attribute has.
#     OUPUT:
#     - returns histogram of mean_degrees
#     '''
#     # if G and attr:
#     mean_degrees = edge_analysis(G, attr = 'intermediary')
#     plt.hist(mean_degrees, bins = 50)
#     plt.xlabel('The Mean Degree Per Node For {}'.format(attr))
#     plt.ylabel('Frequency of Mean Degree')
#     # else:
#     #     plt.hist(mean_degeers, bins = 20)

def k_mean_cluster(G):
    A = nx.adjacency_matrix(G)
    D = np.diag(np.ravel(np.sum(A,axis=1)))
    L = D - A
    # compute eigenvalues/eigenvectors using eig
    # 'l' is the evals and 'U' is the evects
    l, U = la.eigh(np.cov(L))
    f = U[:,1]
    labels = np.ravel(np.sign(f))
    pass

def plot_hist_size_partition():
    unique_size = len(unique(list(partition.values())))
    plt.hist(partition.values(), bins = unique_size)
    plt.xlabel('The Number of Node In Partition')
    plt.ylabel('Frequency of Partition Size')
    plt.title('Distribution of Partition Size in Nodes')

if __name__ == '__main__':
    # F, all_nodes = load_clean_data() # Import and clean
    # ego = build_subgraph(F, all_nodes ) # Create subgroup
    # GeneralGraph(ego, filename = "ego_3") #Generages image in Gephi
    G, all_nodes = lg.load_clean_data() # Import and clean
    ego = lg.build_subgraph(G, all_nodes ) # Create subgroup
    lg.GeneralGraph(ego, filename = "consolidation_nulls") #Generages image in Gephi
    # f = filter_nodess_by_degree(ego, all_nodes, k=3)
    # lg.GeneralGraph(ego, filename = "ego_filter_k_3")
    # # DiGraphMatcher.subgraph_is_isomorphic(f,ego)
    Q = msr.my_louvian_modularity(ego)

    df_centralities = msr.comparing_centralities(ego)
    md_centralities = pandas_df_to_markdown_table(df_centralities)

    dgre_connectivity_dict, mean_degrees = msr.edge_analysis(ego, attr = 'intermediary')
    msr.plot_hist_avg_degrees(ego, attr = 'intermediary')
