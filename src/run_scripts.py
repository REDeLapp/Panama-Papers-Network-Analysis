import networkx as nx
import load_and_graph2 as lg
import measuring2 as msr

def filter_nodess_by_degree(G, all_nodes, k):
    '''
    GOAL: this function removes all nodes of a graph G of
    degrees less than or equal to k
    --------------------------------------
    INPUT:
    - G, instantiated networkx graphs
    - 'all_nodes', is the concatenation of all node lists into one dataframe
    - k, the number of degree on which you want to filter
    OUPUT: returns a networkx graph
    '''
    # filter nodes with degree less than k
    nodes = all_nodes.reindex(G)
    nodes = nodes[~nodes.index.duplicated()]
    f = nx.Graph()
    fedges = filter(lambda x: G.degree()[x[0]] > k and G.degree()[x[1]] > k, G.edges())
    f.add_edges_from(fedges)

    # reattach attribute
    nx.set_node_attributes(f, nodes["country_codes"], "cc")
    nx.set_node_attributes(f, nodes["type"], "ty")
    nx.set_node_attributes(f, nodes["name"], "nm")
    # get rid of null and turn the list into a dictionary
    valid_names = nodes[nodes["name"].notnull()]["name"].to_dict()
    f = nx.relabel_nodes(f, nodes[nodes.name.notnull()].name)
    # ego = nx.relabel_nodes(ego, nodes[nodes.address.notnull()
    #                                 & nodes.name.isnull()].address)
    nx.relabel_nodes(f, valid_names)
    return f

def edge_analysis(G, attr = 'intermediary'):
    '''
    Find the average number of degree a node with a certain attribute
    '''
    # node_inter = filter(lambda n, d: d['type'] == 'intermediary', G.nodes(data=True))
    # intermediaries = pd.read_csv('/Users/rdelapp/Galvanize/DSI_g61/capstone/panama_papers/data/csv_panama_papers_2018-02-14/panama_papers_nodes_intermediary.csv', index_col = "node_id")
    # idx_nodes = all_nodes[all_nodes.type == 'intermediary'].index
    # foo = intermediaries.name
    nodes_of_interest = node_attr(G, attr = 'intermediary')
    avg_degree_dict = nx.average_degree_connectivity(G, nodes = nodes_of_interest)
    return avg_degree_dict

def node_attr(G, attr = 'intermediary'):
    return [x for x,y in G.nodes(data=True) if y['ty'] == attr]

def pandas_df_to_markdown_table(df):
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))

if __name__ == '__main__':
    G, all_nodes = lg.load_clean_data() # Import and clean
    ego = lg.build_subgraph(G, all_nodes ) # Create subgroup
    # lg.GeneralGraph(ego, filename = "ego_no_filter") #Generages image in Gephi
    #
    # f = filter_nodess_by_degree(ego, all_nodes, k=3)
    # lg.GeneralGraph(ego, filename = "ego_filter_k_3")
    # # DiGraphMatcher.subgraph_is_isomorphic(f,ego)

    edge_anlysis(ego, attr = 'intermediary')

    # nodesAt5 = [p, for (p,d) in ego.nodes(data=True) if d['ty']== 'intermediary']
    #  # ...:         for (p, d) in ego.nodes(data=True):
    #  # ...:             if d['ty'] == 'intermediary':
    #  # ...:                 nodesAt5.append(p)
    nodes_of_interest = [x for x,y in ego.nodes(data=True) if y['ty']== 'intermediary' ]
    nx.average_degree_connectivity(ego, nodes = nodes_of_interest)
    foo = ego.degree(nodes_of_interest)
    bar = [v  for k,v in foo]
