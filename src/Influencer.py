class Influencer(Persons):
    def _init_(self, id):
        self.id = id
        self.i = r.random()
        self.a = 1 ## opinion is strong and immovavle

    def step(self):
        pass

if __name__ == '__main__':

    Influencer = 2
    connections = 4
    ## Add the influencer to the network and connect each to the 3 other nodes
    for i in range(influencers):
        inf = Influencer ("Inf" = str(i))
        for x in range(connections):
            g.add_edge(r.choice(g.nodes()), inf)
    ## collect new attributes data, print it to the terminal and plot itself
    col [n.a for n in g.nodes()]
    print(col)
    plot.plot(col)
