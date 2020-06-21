"""
Script for getting scores data (mostly problem specific but I generalized a few concepts)

:copyright: (c) 2020, Joeri Nicolaes
:license: BSD license, see LICENSE for more details.
"""
import argparse
import numpy as np
import pandas as pd
from peerscore import PeerScoresGraph, Peer


def read_df(fn):
    """
    Reads xlsx scores into OrderedDict pandas dataframe
    Contains specific pre-processing for scores file from KUL-H02A5a

    :param fn: str
    :return: OrderedDict object returned by pandas (one df for each sheet)
    """
    def change_name(name_str):
        return name_str.replace('Ontvangen door ', '')
    df = pd.read_excel(fn, sheet_name=None)
    for k in df.keys():
        # NA is represented as 6 > change to np.nan
        df[k].replace(6, np.nan)
        df[k].Naam = df[k].Naam.apply(change_name)
        cols = [c for c in df[k].columns.values if 'B' in c]
        [df[k].rename(columns={c: df[k].Naam[df[k].Id == c].values[0]}, inplace=True) for c in cols]
        df[k] = df[k].drop(columns=['Id'])
    full_df = pd.concat([df[k] for k in df.keys()], sort=False)
    return full_df.set_index('Naam')    #.sort_index(axis=1)


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
    df = read_df(fn)
    return [Peer(peer_unique_id=i, peer_friendly_name=n) for i,n in enumerate(df.index.values)], df


def tuple_series_index_and_values(s):
    return list(zip(s, s.index))


def get_scores_from_peer_in_database(peer_friendly_name, df):
    """
    Get scores given by peer_id in database
    This function is specific to your scores database file and should be overwritten !!!!

    Canonical implementation (score and Peer() should initialized from data file):
        def get_scores_from_peer_in_database(peer_id, fn):
            return [(score, Peer())]

    :param peer_friendly_name: str
    :param df: pandas.DataFrame
    :return: list, each entry is a tuple (score, to_peer) with
        score (int or float are supported)
    """
    scores_given_by_peer = df[peer_friendly_name][~df[peer_friendly_name].isnull()]
    return tuple_series_index_and_values(scores_given_by_peer)


def get_peerscores_from_database(fn):
    """
    Get the scores for all peers from the scores database in fn
    a peer score is an int or float value given by one peer to another peer

    :param fn: str, scores database file
    :return:
    """
    # 1. initialize graph with Peer objects as nodes
    inputscores_graph = PeerScoresGraph()
    peers, df = get_peers_from_database(fn)
    for peer_obj in peers:
        inputscores_graph.add_peer(peer_obj)
    # 2. update graph with scores as edges
    for from_peer in inputscores_graph.nodes():
        scores_list = get_scores_from_peer_in_database(from_peer.peer_friendly_name, df)
        for score, to_peer in scores_list:
            inputscores_graph.add_score(score, from_peer, Peer(peer_unique_id=-1, peer_friendly_name=to_peer))
    return inputscores_graph


def main(scores_fn):
    # peer_inputscores_graph = get_peerscores_from_database(scores_fn)
    data_matrix = read_df(scores_fn)
    print(data_matrix.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get peer scores from given file')
    parser.add_argument('scores_file', metavar='scores_fn', help='string path to scores filename')
    args = parser.parse_args()

    if main(args.scores_file):
        sys.exit("200 OK ")
    else:
        sys.exit("400 Bad Request")




