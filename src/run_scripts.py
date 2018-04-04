import networkx as nx
import load_and_graph2 as lg
import measuring2 as msr

def filter_nodess_by_degree(G, all_nodes, k):
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

def edge_anlysis(G, attr = 'intermediary'):
    ''' Find the average number of degree a node with a certain attribute
    '''
    node_inter = filter(lambda n, d: d['type'] == 'intermediary', G.nodes(data=True))
    nx.average_degree_connectivity(ego, nodes = node_inter)
    pass

if __name__ == '__main__':
    G, all_nodes = lg.load_clean_data() # Import and clean
    ego = lg.build_subgraph(G, all_nodes ) # Create subgroup
    lg.GeneralGraph(ego, filename = "ego_no_filter") #Generages image in Gephi

    f = filter_nodess_by_degree(ego, all_nodes, k=3)
    lg.GeneralGraph(ego, filename = "ego_filter_k_3")
