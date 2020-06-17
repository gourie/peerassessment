"""
Peer scores data structure

:copyright: (c) 2020, Joeri Nicolaes
:license: BSD license, see LICENSE for more details.
"""
from graph import WeightedGraph


class PeerScoresGraph(WeightedGraph):
    """
    Peers scores data structure

    The peer scores data structure is constructed in two steps:
    1. a graph is initialized with Peer objects as nodes
    2. the graph is updated with scores as edges

    The graph can be queried for scores between peers (edges).
    The get_data_matrix method is the API to build the calibration model

    """

    def __init__(self):
        """
        Constructor

        """
        super(PeerScoresGraph, self).__init__(*args, **kwargs)

    def add_peer(self, peer):
        self.edges[peer] = {}

    def add_score(self, score, from_peer, to_peer):
        assert (type(score) == int or type(score) == float)
        self.edges[from_peer][to_peer] = score

    def get_data_matrix(self):
        """
        Get data matrix, API to build calibration model

        :return:
        """
        return


class Peer(object):
    """
    Peer capable of giving and/or receiving scores
    """

    def __init__(self, peer_unique_id, peer_friendly_name=None):
        """
        Constructor

        :param peer_unique_id: str, unique_id to identify peer
        """
        self.peer_unique_id = str(peer_unique_id)
        self.peer_friendly_name = peer_friendly_name

    @property
    def peer_unique_id(self):
        return self.__peer_unique_id

    @peer_unique_id.setter
    def peer_unique_id(self, value):
        assert isinstance(value, str)
        self.__peer_unique_id = value

    @property
    def peer_friendly_name(self):
        return self.__peer_friendly_name

    @peer_friendly_name.setter
    def peer_friendly_name(self, value):
        self.__peer_friendly_name = value


class PeerScore(object):
    """
    Assessment score between two peers
    Scores are modeled as int or float values (based on the type of score argument)

    TODO: OBSOLETE when using graph structure> considering removing

    """

    def __init__(self, score, from_peer, to_peer):
        """
        Constructor

        :param score: int or float, score value
        :param from_peer: Peer object
        :param to_peer: Peer object
        """
        self.score = score
        self.score_type = type(score)
        assert (self.score_type == int or self.score_type == float)
        self.from_peer = from_peer
        self.to_peer = to_peer

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, value):
        self.__score = value

    @property
    def from_peer(self):
        return self.__from_peer

    @from_peer.setter
    def from_peer(self, value):
        self.__from_peer = value

    @property
    def to_peer(self):
        return self.__to_peer

    @to_peer.setter
    def to_peer(self, value):
        self.__to_peer = value