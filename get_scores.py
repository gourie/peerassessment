"""
Script for getting scores data (mostly problem specific but I generalized a few concepts)

:copyright: (c) 2020, Joeri Nicolaes
:license: BSD license, see LICENSE for more details.
"""
import ArgumentParser
from peerscore import PeerScoresGraph, Peer


def get_peers_from_database(fn):
    """
    Get peers from scores database in fn
    This function is specific to your scores data file and should be overwritten!!!!

    Canonical implementation (Peer() should initialized from data file):
        def get_peers_from_database(fn):
            return [Peer()]

    :param fn: str
    :return: list of Peer objects
    """
    return [Peer()]


def get_scores_from_peer_in_database(peer_id, fn):
    """
    Get scores given by peer_id in database
    This function is specific to your scores database file and should be overwritten !!!!

    Canonical implementation (score and Peer() should initialized from data file):
        def get_scores_from_peer_in_database(peer_id, fn):
            return [(score, Peer())]

    :param peer_id: str
    :param fn: str, scores database file
    :return: list, each entry is a tuple (score, to_peer) with
        score (int or float are supported)
    """
    return [(3, Peer())]


def get_peerscores_from_database(fn):
    """
    Get the scores for all peers from the scores database in fn
    a peer score is an int or float value given by one peer to another peer

    :param fn: str, scores database file
    :return:
    """
    # 1. initialize graph with Peer objects as nodes
    inputscores_graph = PeerScoresGraph()
    for peer_obj in get_peers_from_database(fn):
        inputscores_graph.add_peer(peer_obj)
    # 2. update graph with scores as edges
    for peer_obj in inputscores_graph.nodes():
        scores_list = get_scores_from_peer_in_database(peer_obj.peer_unique_id, fn)
        for score, to_peer in scores_list:
            inputscores_graph.add_score(score, peer_obj, to_peer)
    return inputscores_graph


def main(scores_fn):
    peer_inputscores_graph = get_peerscores_from_database(scores_fn)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get peer scores from given file')
    parser.add_argument('scores_file', metavar='scores_fn', help='string path to scores filename')
    args = parser.parse_args()

    if main(args.scores_file):
        sys.exit("200 OK ")
    else:
        sys.exit("400 Bad Request")




