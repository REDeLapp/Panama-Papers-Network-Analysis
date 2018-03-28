# Panama-Papers-Network-Analysis

## The Project
Social network analysis of the Panama Papers, with a specific emphasis in Saudi Arabia and, maybe, Jordan.

## Scope
To Do’s: 
- (5 days) Analyze Network
- (2 days) Visualization
- (2 days) Background research  (i.e. Who are these people?)
- (1 day) Clean Data


## The Data
The Panama  Papers are a set of 11.5 million document leaks from Panamanian law company ”Mossack Fonseca”, which provides
information on approximately 360,000 businesses and individuals in more than 200 countries linked to offshore structures and covering a time period of nearly 40 years, from 1977 to 2016.

The  ”ICIJ  Offshore”  database,  presents  the  network  of  relationships  between  companies  and  individual  people  with  offshore  companies based in tax havens. Consists in a directed and unweighted network based on commercial  registration  of  all  types  of  companies  involved  in  the  scandal  and the existing relations type, which are:

* "director of” - referring to the person appointed to the company’s management;

* ”address” - through which was possible to establish the country origin of the company;

* ”shareholder of” - if it holds a stake in an offshore company;

* ”intermediary of” - if it mediates companies in access to offshores;

* ”similar of” - if the company is related to another company, among other attributes.

How The Data Is Structured:

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
