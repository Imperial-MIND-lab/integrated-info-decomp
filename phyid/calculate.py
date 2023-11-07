# -*- coding: utf-8 -*-
"""Calculate the information decompositions."""

import numpy as np
from .measures import (
    local_entropy_mvn,
    local_entropy_binary,
    redundancy_mmi,
    double_redundacy_mmi,
    redundancy_ccs,
    double_redundacy_ccs,
)
from .utils import (
    PhiID_atoms_abbr,
    _binarize
)


def _get_entropy_four_vec(X, kind):
    if kind == "gaussian":
        X_cov = np.cov(X)
        X_mu = np.mean(X, axis=1)
        def _h(idx):
            return local_entropy_mvn(X[idx].T, X_mu[idx], X_cov[np.ix_(idx, idx)])
    elif kind == "discrete":
        def _h(idx):
            return local_entropy_binary(X[idx, :])
    else:
        raise ValueError("kind must be one of 'gaussian' or 'discrete'")

    p1, p2, t1, t2 = [0, 1, 2, 3]

    h_res = {
        "h_p1": _h([p1]),
        "h_p2": _h([p2]),
        "h_t1": _h([t1]),
        "h_t2": _h([t2]),
        "h_p1p2": _h([p1, p2]),
        "h_t1t2": _h([t1, t2]),
        "h_p1t1": _h([p1, t1]),
        "h_p1t2": _h([p1, t2]),
        "h_p2t1": _h([p2, t1]),
        "h_p2t2": _h([p2, t2]),
        "h_p1p2t1": _h([p1, p2, t1]),
        "h_p1p2t2": _h([p1, p2, t2]),
        "h_p1t1t2": _h([p1, t1, t2]),
        "h_p2t1t2": _h([p2, t1, t2]),
        "h_p1p2t1t2": _h([p1, p2, t1, t2]),
    }
    return h_res


def _get_coinfo_four_vec(h_res):
    I_res = {
        "I_xytab": h_res["h_p1p2"] + h_res["h_t1t2"] - h_res["h_p1p2t1t2"],
        "I_xta": h_res["h_p1"] + h_res["h_t1"] - h_res["h_p1t1"],
        "I_xtb": h_res["h_p1"] + h_res["h_t2"] - h_res["h_p1t2"],
        "I_yta": h_res["h_p2"] + h_res["h_t1"] - h_res["h_p2t1"],
        "I_ytb": h_res["h_p2"] + h_res["h_t2"] - h_res["h_p2t2"],
        "I_xyta": h_res["h_p1p2"] + h_res["h_t1"] - h_res["h_p1p2t1"],
        "I_xytb": h_res["h_p1p2"] + h_res["h_t2"] - h_res["h_p1p2t2"],
        "I_xtab": h_res["h_p1"] + h_res["h_t1t2"] - h_res["h_p1t1t2"],
        "I_ytab": h_res["h_p2"] + h_res["h_t1t2"] - h_res["h_p2t1t2"],
    }
    return I_res


def _get_redundancy_four_vec(redundancy, I_res):
    if redundancy == "MMI":
        redundancy_func = redundancy_mmi
    elif redundancy == "CCS":
        redundancy_func = redundancy_ccs
    else:
        raise ValueError("redundancy must be one of 'MMI' or 'CCS'")

    R_res = {
        "R_xyta": redundancy_func(I_res["I_xta"], I_res["I_yta"], I_res["I_xyta"]),
        "R_xytb": redundancy_func(I_res["I_xtb"], I_res["I_ytb"], I_res["I_xytb"]),
        "R_xytab": redundancy_func(I_res["I_xtab"], I_res["I_ytab"], I_res["I_xytab"]),
        "R_abtx": redundancy_func(I_res["I_xta"], I_res["I_xtb"], I_res["I_xtab"]),
        "R_abty": redundancy_func(I_res["I_yta"], I_res["I_ytb"], I_res["I_ytab"]),
        "R_abtxy": redundancy_func(I_res["I_xyta"], I_res["I_xytb"], I_res["I_xytab"]),
    }
    return R_res


def _get_double_redundancy_four_vec(redundancy, calc_res):
    I_res = calc_res["I_res"]
    R_res = calc_res["R_res"]

    if redundancy == "MMI":
        rtr_input_list = [
            I_res["I_xta"],
            I_res["I_xtb"],
            I_res["I_yta"],
            I_res["I_ytb"],
        ]
        double_redundancy_func = double_redundacy_mmi
    elif redundancy == "CCS":
        double_coinfo = - I_res["I_xta"] - I_res["I_xtb"] - I_res["I_yta"] - I_res["I_ytb"] + \
            I_res["I_xtab"] + I_res["I_ytab"] + I_res["I_xyta"] + I_res["I_xytb"] - I_res["I_xytab"] + \
            R_res["R_xyta"] + R_res["R_xytb"] - R_res["R_xytab"] + \
            R_res["R_abtx"] + R_res["R_abty"] - R_res["R_abtxy"]
        rtr_input_list = [
            I_res["I_xta"],
            I_res["I_xtb"],
            I_res["I_yta"],
            I_res["I_ytb"],
            double_coinfo
        ]
        double_redundancy_func = double_redundacy_ccs
    else:
        raise ValueError("redundancy must be one of 'MMI' or 'CCS'")

    rtr = double_redundancy_func(rtr_input_list)
    return rtr


def _get_atoms_four_vec(calc_res):

    I_res = calc_res["I_res"]
    R_res = calc_res["R_res"]
    rtr = calc_res["rtr"]

    knowns = np.c_[
        rtr,
        R_res["R_xyta"], R_res["R_xytb"], R_res["R_xytab"],
        R_res["R_abtx"], R_res["R_abty"], R_res["R_abtxy"],
        I_res["I_xta"], I_res["I_xtb"], I_res["I_yta"], I_res["I_ytb"],
        I_res["I_xyta"], I_res["I_xytb"], I_res["I_xtab"], I_res["I_ytab"], I_res["I_xytab"]
    ]

    knowns_to_atoms_mat = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rtr
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxyta
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytb
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytab
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rabtx
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Rabty
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Rabtxy
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixta
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixtb
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Iyta
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # Iytb
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # Ixyta
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Ixytb
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixtab
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # Iytab
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ixytab
    ]

    atoms_mat = np.linalg.solve(knowns_to_atoms_mat, knowns.T)
    atoms_res = {_: atoms_mat[i, :] for i, _ in enumerate(PhiID_atoms_abbr)}
    return atoms_res


def calc_PhiID(src, trg, tau, kind="gaussian", redundancy="MMI"):
    """
    Calculate the information decompositions.

    This function calculates the information decompositions of two time series.

    Parameters
    ----------
    src : (N,) array_like
        Source time series.
    trg : (N,) array_like
        Target time series.
    tau : int
        Time lag.
    kind : str, optional
        PhiID kind to calculate, by default "gaussian".
    redundancy : str, optional
        Redundancy measure to use, by default "MMI".

    Returns
    -------
    atoms_res : dict
        Dictionary with the atoms of the decomposition.
    calc_res : dict
        Dictionary with the intermediate calculations.
    """
    # check input vectors
    assert len(src) == len(trg)

    # construct four vectors
    src_past, src_future = src[:-tau], src[tau:]
    trg_past, trg_future = trg[:-tau], trg[tau:]

    # check kind
    if kind == "gaussian":
        X = np.c_[src_past, trg_past, src_future, trg_future].T  # ["sp", "tp", "sf", "tf"]
        X_norm = X / np.std(X, axis=1, ddof=1, keepdims=True)
        X_input = X_norm
    elif kind == "discrete":
        X = np.c_[
            _binarize(src_past),
            _binarize(trg_past),
            _binarize(src_future),
            _binarize(trg_future)
        ].T
        X_input = X
    else:
        raise ValueError("kind must be one of 'gaussian' or 'discrete'")

    h_res = _get_entropy_four_vec(X_input, kind=kind)
    I_res = _get_coinfo_four_vec(h_res)
    R_res = _get_redundancy_four_vec(redundancy, I_res)
    calc_res = {
        "h_res": h_res,
        "I_res": I_res,
        "R_res": R_res
    }
    rtr = _get_double_redundancy_four_vec(redundancy, calc_res)
    calc_res["rtr"] = rtr
    atoms_res = _get_atoms_four_vec(calc_res)

    return atoms_res, calc_res

