"""
Script for getting scores data (mostly problem specific but I generalized a few concepts)

:copyright: (c) 2020, Joeri Nicolaes
:license: BSD license, see LICENSE for more details.
"""
import argparse
from itertools import product
import numpy as np
import os
import pandas as pd
import sys
from typing import Dict, List, Tuple


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
        df[k] = df[k].replace(6, np.nan)
        df[k].Naam = df[k].Naam.apply(change_name)
        cols = [c for c in df[k].columns.values if 'B' in c]
        [df[k].rename(columns={c: df[k].Naam[df[k].Id == c].values[0]}, inplace=True) for c in cols]
        df[k] = df[k].drop(columns=['Id'])
    full_df = pd.concat([df[k] for k in df.keys()], sort=False)
    return full_df.set_index('Naam')    #.sort_index(axis=1)


def get_sigma_q(peer_scores: np.ndarray, sigma_b_hat, sigma_s_hat) -> Tuple:
    """
    Get sigma_q

    First all self-scores are deleted (not used)
        diag(P) := 0

    :param peer_scores: np.ndarray with peer scores
    :return: tuple (y, K_hat)
    """
    # build data matrix
    # scores_tuples = np.argwhere(~np.isnan(peer_scores))
    # scores_list = np.where(~np.isnan(peer_scores))
    # y = peer_scores[np.where(~np.isnan(peer_scores))]
    # N = y.shape[0]
    y, N, scores_tuples, scores_list = get_scores(peer_scores)
    K_hat = get_K(N, scores_list, scores_tuples, sigma_q=1, sigma_b=sigma_b_hat, sigma_s=sigma_s_hat)
    sigma_q = 1/N * np.dot(np.dot(y.T, np.linalg.inv(K_hat)), y)
    return sigma_q, K_hat, y


def get_scores(peer_scores):
    scores_tuples = np.argwhere(~np.isnan(peer_scores))
    scores_list = np.where(~np.isnan(peer_scores))
    y = peer_scores[np.where(~np.isnan(peer_scores))]
    N = y.shape[0]
    return y, N, scores_tuples, scores_list


def get_K(N: int, scores_list: List, scores_tuples: Tuple, sigma_q: float, sigma_b: float, sigma_s: float) -> np.ndarray:
    """
    Get covariance matrix K with
        K[i,j;k,l] = delta(i,k)*sigma_q  + delta(j,l)*sigma_b + delta(i,k)*delta(j,l)*sigma_s

    :param N: int, dim
    :param scores_list:
    :param scores_tuples:
    :param sigma_q:
    :param sigma_b:
    :param sigma_s:
    :return: np.ndarray of shape (N, N)
    """
    K = np.zeros((N, N))
    # K_diag_indices = np.diag_indices(N)     # np.stack(2 * [scores_list[0]], axis=-1)
    # _, c = np.unique(scores_list[0], return_counts=True)

    # delta(i,k) = 1
    for i in np.unique(scores_list[0]):
        matching_i = np.where(scores_list[0] == i)[0]
        for match_i in matching_i:
            K[scores_tuples[match_i]] += sigma_q
    # delta(i,k) = delta(j,l) = 1
    K_diag_indices = np.diag_indices(N)
    K[K_diag_indices] += sigma_s
    # for score_index, _ in enumerate(scores_tuples):
    #     K[score_index, score_index] += sigma_s_hat
    # delta(j,l) = 1
    for j in np.unique(scores_list[1]):
        matching_j = np.where(scores_list[1] == j)[0]
        for match_j in matching_j:
            K[scores_tuples[match_j]] += sigma_b
    return K


def train_calibration_model(peer_scores):
    """
    Train calibration model

    Adapted from Lawrence / NeurIPS reviewer calibration (see [here](https://inverseprobability.com/2014/08/02/reviewer-calibration-for-nips))

    Our assumption is that the score from the \(j\)th peer for the \(i\)th peer is given by
    \[y_{i,j} = f_i + b_j + \epsilon_{i, j}\]

    where
    \(f_i\) is the objective quality of peer's contribution \(i\)
    \(b_j\) is an offset associated with peer \(j\)
    \(\epsilon_{i,j}\) is a subjective quality estimate which reflects how a specific peer’s opinion differs from other peers (such differences in opinion may be due to differing expertise or perspective).
    The underlying ‘objective quality’ of the peer contribution is assumed to be the same for all peers and the peer score offset is assumed to be the same for all peers.

    :param data_tuple: y, deltaa
    :return:
    """
    grid_sigma = np.linspace(0.1, 4, 20)
    l_sigma_q = []
    l_likelihood = []
    l_grid = []
    for sigma_b_hat, sigma_s_hat in product(grid_sigma, repeat=2):
        sigma_q, K_hat, y = get_sigma_q(peer_scores, sigma_b_hat, sigma_s_hat)
        l_sigma_q.append(sigma_q)
        l_likelihood.append(get_neg_log_likelihood(y, K_hat, sigma_q))
        l_grid.append((sigma_b_hat * sigma_q, sigma_s_hat * sigma_q))
    lowest_lik_ind = np.argmin(l_likelihood)
    return l_sigma_q[lowest_lik_ind], l_grid[lowest_lik_ind]


def get_neg_log_likelihood(y, K_hat, sigma_q, epsilon=1e-7):
    N = y.shape[0]
    K_hat_inverse = np.linalg.inv(K_hat)
    K_hat_determ = np.linalg.det(K_hat)
    return N / 2 * np.log(2 * np.pi * sigma_q) + 1 / 2 * np.log(K_hat_determ + epsilon) + 1 / 2 * sigma_q * np.dot(
        np.dot(y.T, K_hat_inverse), y)


def get_calibrated_peerscores(peer_scores, model_params: Dict):
    """
    Get the calibrated scores for all peers

    :return: calibrated scores
    """
    y, N, scores_tuples, scores_list = get_scores(peer_scores)
    K = get_K(N, scores_list, scores_tuples, model_params['sigma_q'], model_params['sigma_b'], model_params['sigma_s'])
    K_s = get_K(N, scores_list, scores_tuples, model_params['sigma_q'], 0, model_params['sigma_s'])
    mu_s = np.dot(np.dot(K_s, np.linalg.inv(K)), y)
    cov_s = K_s - np.dot(np.dot(K_s, np.linalg.inv(K)), K_s)
    return y, mu_s, scores_tuples


def get_out_df(y: np.ndarray, mu_s: np.ndarray, scores_tuples: np.ndarray, scores_df_index: pd.Index, mean_score: float, correction_range: Tuple) -> Tuple:
    """
    Get output dataframe

    :param y: original scores, shape (N,)
    :param mu_s: calibrated scores correcting for bias of peers given scores, shape (N,)
    :param scores_tuples: (student_receiving_score, peer_giving_score), shape (N,2)
    :param scores_df_index: index of students receiving scores
    :return: pd.DataFrame summary, pd.DataFrame details
        df/sheet=summary, index of students receiving scores
            index: students receiving score
            columns: average_score, average_calibrated_score, individual_correction
        df/sheet=details, index of students receiving/giving scores
            index: students receiving score
            columns: calibrated scores
    :param mean_score: mean score (depending on the scoring scheme)
    :param correction_range: (max_negative_correction, max_positive_correction)
    """
    y_reord = np.zeros((len(scores_df_index),len(scores_df_index)))
    y_calibrated = np.zeros((len(scores_df_index), len(scores_df_index)))
    for scores_index, tup in enumerate(scores_tuples):
        y_reord[tup[0], tup[1]] = y[scores_index]
        y_calibrated[tup[0], tup[1]] = mu_s[scores_index]
    y_reord[y_reord == 0] = np.nan
    y_calibrated[y_calibrated == 0] = np.nan
    df_details = pd.DataFrame(y_calibrated, index=scores_df_index)
    [df_details.rename(columns={c: df_details.index[c]}, inplace=True) for c in range(len(df_details.columns))]
    df_summary = pd.DataFrame(np.nanmean(y_reord, axis=-1).T, index=scores_df_index)
    df_summary.rename(columns={0: 'average_score'}, inplace=True)
    df_summary['average_calibrated_score'] = df_details.mean(axis=1).values.T
    df_summary['individual_correction'] = normalize_and_round_corrections(
        df_summary['average_calibrated_score'] - mean_score, correction_range)
    return df_details, df_summary


def normalize_and_round_corrections(average_calibrated_score, correction_range):
    """
    Normalizes point corrections to given range and rounds to closest 0.25

    :param average_calibrated_score:
    :param correction_range:
    :return:
    """
    scale = (np.max(correction_range) - np.min(correction_range)) / (
                np.max(average_calibrated_score) - np.min(average_calibrated_score))
    shift = np.max(correction_range) - scale * np.max(average_calibrated_score)
    return np.round((shift + scale * average_calibrated_score) * 4) / 4


def main(scores_fn, mean_score: float=3., correction_range=(-1, 1)):
    scores_df = read_df(scores_fn)
    # ignore self-scores on diagonal
    peer_scores = scores_df.values - mean_score
    diag_indices = np.diag_indices(peer_scores.shape[0])
    peer_scores[diag_indices] = np.nan
    sigma_q, tuple_sigma_b_sigma_s = train_calibration_model(peer_scores)
    model_params = dict(sigma_q=sigma_q,
                        sigma_b=tuple_sigma_b_sigma_s[0],
                        sigma_s=tuple_sigma_b_sigma_s[1]
                        )
    y, mu_s, scores_tuples = get_calibrated_peerscores(peer_scores, model_params)
    df_details, df_summary = get_out_df(y + mean_score, mu_s + mean_score, scores_tuples, scores_df.index, mean_score,
                                        correction_range)
    output_fn = os.path.join(os.getcwd(), 'calibrated_scores.xlsx')
    with pd.ExcelWriter(output_fn) as writer:
        df_summary.to_excel(writer, sheet_name='summary')
        df_details.to_excel(writer, sheet_name='details')
    return output_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get peer scores from given file')
    parser.add_argument('--input', metavar='scores_fn', help='string path to scores filename')
    parser.add_argument('--mean_score', metavar='mean_score', type=float,
                        help='mean score, e.g. if students give scores 1 to 5, mean_score=3')
    parser.add_argument('--max_negative_correction', metavar='max_negative_correction', type=float,
                        help='max_negative_correction applied to the students points')
    parser.add_argument('--max_positive_correction', metavar='max_positive_correction', type=float,
                        help='max_positive_correction applied to the students points')
    args = parser.parse_args()

    output_fn = main(args.input, args.mean_score, (args.max_negative_correction, args.max_positive_correction))
    if output_fn:
        sys.exit("Calibrated scores saved to file %s" % output_fn)
    else:
        sys.exit("400 Bad Request")




