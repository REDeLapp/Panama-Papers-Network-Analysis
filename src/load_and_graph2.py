import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# https://www.occrp.org/en/panamapapers/database

def load_clean_data():
    '''
    GOAL: This function builds the edges and nodes from the ICIJ open source data base
    of the Panama Papers. These files are concatenated and used to build networkX.Pandas.DataFrame(F), combines all
    the nodes from the lists into one data frame, and performs some data cleaning.
    ----------------------------
    INPUT: None
    OUPUT:
    - 'F' is the NetworkX graph from Pandas DataFrame.
    - 'all_nodes', is the concatenation of all node lists into one dataframe
    '''
    # Read the edge list and convert it to network
    edges = pd.read_csv('''../data/csv_panama_papers_2018-02-14/panama_papers_edges.csv''')
    edges = edges[edges["TYPE"] != "registrated address"]
    F = nx.from_pandas_dataframe(edges, "START_ID", "END_ID") # Return a graph from Pandas DataFrame.

    # Read node lists
    officers       = pd.read_csv("../data/csv_panama_papers_2018-02-14/panama_papers_nodes_officer.csv", index_col = "node_id")
    intermediaries = pd.read_csv("../data/csv_panama_papers_2018-02-14/panama_papers_nodes_intermediary.csv", index_col = "node_id")
    entities       = pd.read_csv("../data/csv_panama_papers_2018-02-14/panama_papers_nodes_entity.csv", index_col = "node_id")

    # Combine the node lists into one dataframe
    officers["type"] = "officer"
    intermediaries["type"] = "intermediary"
    entities["type"] = "entity"
    #Concatentate
    all_nodes = pd.concat([officers, intermediaries, entities])

    # Do some cleanup of names
    all_nodes["name"] = all_nodes["name"].str.upper().str.replace(' ','_')

    # Ensure that all "Bearers" do not become a single node
    all_nodes["name"].replace(
        to_replace = [r"MRS?\.\s+", r"\.", r"\s+", "LIMITED", "THE BEARER", "BEARER", "BEARER 1", "EL PORTADOR", "AL PORTADOR"],
        value = ["","", "", "LTD", np.nan, np.nan, np.nan, np.nan, np.nan],
        inplace = True, regex = True)

    # all_nodes['type'].dropna

    # clear NAN's
    # if "ISSUES OF:" in F:
    #     F.remove_node("ISSUES OF:")
    #
    # if "" in F:
    #     F.remove_node("")
    return F, all_nodes



def build_subgraph(F, all_nodes, filename):
    '''
    GOAL: This function build the subgraph of the notes_of_interest, creates the graph,
    and then saves the figure.
    -------------------------------
    INPUT:
        - 'F' is the NetworkX graph from Pandas DataFrame.
        - 'all_nodes', is the concatenation of all node lists into one dataframe.
        - 'file_name' as string that is name you wnat to save the picture.
        - 'cc'... [edit]
    OUPUT:
        - 'ego' is a subgraph of the countries of interest.
    '''

    # We only want to look at Saudi Arabia and, maybe, Jordan
    # CCODES = ["SAU", "JOR"] # country code to be examined in subgraph
    # CCODES = "UZB", "TKM", "KAZ", "KGZ", "TJK"
    CCODES = "SAU", "JOR"
    #seeds = all_nodes[all_nodes["country_codes"] == 'SAU'].index
    seeds = all_nodes[all_nodes["country_codes"].isin(CCODES)].index

    # # Next Computes the shortest path from the node seed to all reachable nodes that
    # # are cutoff hops away and closer.  The function returns a dictionary with the target nodes as keys,
    # # so the keys are the cutoff-neighborhood of the seed.
    nodes_of_interest = set.union(*[set(nx.single_source_shortest_path_length(F, seed, cutoff = 2).keys())
                                   for seed in seeds])
    # Extract the subgraph that contains all the keys for all the dictionaries
    # with all the connecting eges ... and relabel it
    # ego = nx.subgraph(F, nodes_of_interest)
    ego = nx.subgraph(F, nodes_of_interest)
    nodes = all_nodes.reindex(ego)
    # nodes = all_nodes.loc[ego] # <-- Error is coming from here!
    nodes = nodes[~nodes.index.duplicated()] # There are duplicate countru codes on some nodes

    # nx.set_node_attributes(sub_F, "cc", nodes["country_codes"])
    #  Sets node attributes for nodes["country_codes"] from a given value or dictionary of values
    nx.set_node_attributes(ego, nodes["country_codes"], "cc")
    nx.set_node_attributes(ego, nodes["type"], "ty")
    nx.set_node_attributes(ego, nodes["name"], "nm")
    # get rid of null and turn the list into a dictionary
    valid_names = nodes[nodes["name"].notnull()]["name"].to_dict()
    nx.relabel_nodes(ego, valid_names)

    # Save and proceed to gephi!
    # with open ("panama-saujor.grapml", "wb") as ofile:
    # with open ("../figures/panama-{y}.graphml".format(y=filename), "wb") as ofile:
    #     nx.write_graphml(ego, ofile)
    # pass
    return ego

def GernalGraph(ego, filename):
    # Save and proceed to gephi!
    with open ("../figures/panama-{y}.graphml".format(y=filename), "wb") as ofile:
        nx.write_graphml(ego, ofile)
    pass


if __name__ == '__main__':
    F, all_nodes = load_clean_data() # Import and clean
    ego = build_subgraph(F, all_nodes, filename = "SAU-JOR-old" ) # Create subgroup
    GernalGraph(ego, filename = "practice") #Generages image in Gephi
