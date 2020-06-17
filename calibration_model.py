"""
Script for building calibration model on peer scores

Adapted from Lawrence / NeurIPS reviewer calibration (see [here](https://inverseprobability.com/2014/08/02/reviewer-calibration-for-nips))

Our assumption is that the score from the \(j\)th peer for the \(i\)th peer is given by
\[y_{i,j} = f_i + b_j + \epsilon_{i, j}\]

where
\(f_i\) is the objective quality of peer's contribution \(i\)
\(b_j\) is an offset associated with peer \(j\)
\(\epsilon_{i,j}\) is a subjective quality estimate which reflects how a specific peer’s opinion differs from other peers (such differences in opinion may be due to differing expertise or perspective).
The underlying ‘objective quality’ of the peer contribution is assumed to be the same for all peers and the peer score offset is assumed to be the same for all peers.

:copyright: (c) 2020, Joeri Nicolaes
:license: BSD license, see LICENSE for more details.
"""

import ArgumentParser
from peerscore import PeerScoresGraph, Peer


def train_calibration_model(fn):
    """
    set the scores for all peers from the scores database in fn
    a peer score is an int or float value given by one peer to another peer

    :param fn: str
    :return:
    """
    return


def get_calibrated_peerscores():
    """
    Get the calibrated scores for all peers

    :return: PeerScoresGraph with peers as nodes and calibrated scores as edges
    """
    return


def main(scores_fn):
    # 1. get_data_matrix

    # 2. train calibration model

    # 3. get calibrated peer scores

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get peer scores from given file')
    parser.add_argument('scores_file', metavar='scores_fn',help='string path to scores filename')
    args = parser.parse_args()

    if main(args.scores_file):
        sys.exit("200 OK ")
    else:
        sys.exit("400 Bad Request")




