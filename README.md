# Panama-Papers-Network-Analysis

## The Project

## Scope

## The Data

commercial  registration  of  all  types  of  companies  involved  in  the  scandal  and
the existing relations type, which are:

* "director of” - referring to the person appointed to the company’s management;

* ”address” - through which was possible to establish the country origin of the company;

* ”shareholder of” - if it holds a stake in an offshore company;

* ”intermediary of” - if it mediates companies in access to offshores;

* ”similar of” - if the company is related to another company, among other attributes.


* Entity (offshore): company, trust or fund created in a low-tax, offshore
   jurisdiction by an agent.
   
* Officer: person or company who plays a role in an offshore entity.

* Intermediary: a go-between for someone seeking an offshore corporation
  and an offshore service provider - usually a law-firm or a middleman that
  asks an offshore service provider to create an offshore firm for a client.
  
* Address: contact postal address as it appears in the original databases
  obtained by ICIJ
  
| Name          | Type          | Purpose | # of rows | Columns of interest |
| ------------- |:-------------:| -------:|----------:|------------:|
|  edges.csv    |    Edge       |    |     |         |
| addresses.csv |    Nodes      |     |     |         |
| Entities.csv  |    Nodes      |    |      |         |
| Intermediaries|    Nodes      | |  |    |
| Officers.csv  |    Nodes      | Person
  
  ![Nodes and Relationships](https://github.com/REDeLapp/Panama-Papers-Network-Analysis/blob/master/pictures/filename.png)

## Network Analysis Methods

* Clustering Coefficient: the fraction of possible triangles in an egocentric network that contain the ego node and exit. It measures the         undefined for directed graphs

* Bridges: high  betweenness  individuals  are  often  critical  to  collaboration
  across different groups. 
* Modularity: measure aims to identify the nodes that are more densely connected together than to the rest of the network, describing the network structure,i.e., how the network is compartmentalized into sub-networks
* Closeness and harmonic closeness centrality
* Eigenvector centrality
* PageRank

## EDA

## Model

## References
