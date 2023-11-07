# -*- coding: utf-8 -*-
"""Information-theoretic measures."""

import numpy as np
import scipy.stats as sstats
from itertools import product
from collections import Counter


def local_entropy_mvn(x, mu, cov):
    """
    Calculate the local entropy based on multivariate normal distribution.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_dims)
        input data
    mu : array_like, shape (n_dims,)
        multivariate normal distribution mean
    cov : array_like, shape (n_dims, n_dims)
        multivariate normal distribution covariance matrix

    Returns
    -------
    h : array_like, shape (n_samples,)
        local entropy
    """
    return -np.log(sstats.multivariate_normal.pdf(x, mu, cov))


def local_entropy_binary(x):
    """
    Calculate the local entropy based on binary distribution.

    Parameters
    ----------
    x : array_like, shape (n_dims, n_samples)
        input data

    Returns
    -------
    h : array_like, shape (n_samples,)
        local entropy
    """
    if x.ndim == 1:
        x = x[None, :]
    n_dim, n_samp = x.shape
    combs = list(product([0, 1], repeat=n_dim))
    distri = list(zip(*x.tolist()))
    c = Counter(distri)
    p = np.array([c.get(comb, 0) for comb in combs]) / n_samp
    entropy_dict = {comb: -np.log2(p_) for comb, p_ in zip(combs, p)}
    return np.array([entropy_dict[comb] for comb in distri])


def redundancy_mmi(mi_1, mi_2, mi_12):
    """

    Redundancy based on minimum mutual information.

    Parameters
    ----------
    mi_1 : array_like, shape (n_samples,)
        local mutual information
    mi_2 : array_like, shape (n_samples,)
        local mutual information

    Returns
    -------
    redundancy : array_like, shape (n_samples,)
        redundancy
    """
    return mi_1 if np.mean(mi_1) < np.mean(mi_2) else mi_2

def double_redundacy_mmi(mi_lst):
    """

    Double redundancy based on minimum mutual information.

    Parameters
    ----------
    mi_lst : list of array_like, shape (n_samples,)
        list of local mutual information

    Returns
    -------
    redundancy : array_like, shape (n_samples,)
        double redundancy
    """
    mi_mean_lst = [np.mean(_) for _ in mi_lst]
    return mi_lst[np.argmin(mi_mean_lst)]


def redundancy_ccs(mi_1, mi_2, mi_12):
    """
    Redundancy based on common change in surprisal.

    To be implemented.
    """
    c = mi_12 - mi_1 - mi_2
    signs = np.array([np.sign(mi_1), np.sign(mi_2), np.sign(mi_12), np.sign(-c)]).T
    return np.all(signs == signs[:, 0][:, None], axis=1) * (-c)



def double_redundacy_ccs(mi_lst):
    """
    Double redundancy based on common change in surprisal.

    To be implemented.
    """
    signs = np.array([np.sign(_) for _ in mi_lst]).T
    return np.all(signs == signs[:, 0][:, None], axis=1) * mi_lst[-1]