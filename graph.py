"""
Graph data structure

:copyright: (c) 2020, Joeri Nicolaes
:license: BSD license, see LICENSE for more details.
"""


class SimpleGraph(object):
    """
    Simple graph that can be used to encode a directed or undirected graph
    Graph data structure is dict to allow storing this compactly in JSON files
    Edges are stored as lists in a dict structure containing one key for every node (nodes = self.edges.keys())

    example_graph = SimpleGraph()
    example_graph.edges = {
        'A': ['B'],
        'B': ['A', 'C', 'D'],
        'C': ['A'],
        'D': ['E', 'A'],
        'E': ['B']
    }
    example_graph.n_nodes = 5
    example_graph.n_edges = 8
    example_graph.neighbors(['A']) = ['B']

    TODO: add method save_graph(json_file)
    """
    def __init__(self):
        self.edges = {}

    def n_nodes(self):
        return len(self.edges.keys())

    def n_edges(self):
        return sum([len(self.edges[n]) for n in self.edges.keys()])

    def nodes(self):
        """ Returns all nodes in this graph

        :return: list with one entry for every node
        """
        return list(self.edges.keys())

    def neighbors(self, node_id):
        """ Returns all neighbors of a given node_id

        :param node_id: key to identify node
        :return: list with one entry for every neighbor
        """
        return self.edges[node_id]


class WeightedGraph(SimpleGraph):
    """ Graph with weights for every edge
        Graph data structure is dict of dicts to allow storing this compactly in JSON files

    example_graph = WeightedGraph()
    example_graph.edges = {
        'A': {'B':1},
        'B': {'A':2, 'C':1, 'D':3},
        'C': {'A':3},
        'D': {'E':4, 'A':1},
        'E': {'B':2}
    }
    example_graph.n_nodes = 5
    example_graph.n_edges = 8
    example_graph.neighbors(['A']) = ['B']
    example_graph.cost(['A']['B']) = 1
    example_graph.cost(['A']['C']) = None
    """
    def __init__(self):
        super(WeightedGraph, self).__init__()

    def neighbors(self, node_id):
        """ Returns all neighbors of given node_id

        :param node_id: key to identify node
        :return: list of neighbor_ids
        """
        try:
            return self.edges[node_id].keys()
        except KeyError:
            raise

    def cost(self, from_node, to_node):
        """ Returns cost of edge between from_node and to_node

        :param from_node: node key
        :param to_node: node key
        :return: int, cost of edge between from_node and to_node if this edge exists;
                 if no edge exists between these two nodes None is returned
        """
        try:
            return self.edges[from_node][to_node]
        except:
            return None
