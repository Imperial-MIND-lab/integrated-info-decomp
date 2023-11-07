# -*- coding: utf-8 -*-
"""Utilities for phyid."""
import numpy as np


PhiID_atoms_abbr = [
    "rtr", "rtx", "rty", "rts",
    "xtr", "xtx", "xty", "xts",
    "ytr", "ytx", "yty", "yts",
    "str", "stx", "sty", "sts",
]

PhiID_atoms_full = [
    ("Red", "Red"), ("Red", "Un1"), ("Red", "Un2"), ("Red", "Syn"),
    ("Un1", "Red"), ("Un1", "Un1"), ("Un1", "Un2"), ("Un1", "Syn"),
    ("Un2", "Red"), ("Un2", "Un1"), ("Un2", "Un2"), ("Un2", "Syn"),
    ("Syn", "Red"), ("Syn", "Un1"), ("Syn", "Un2"), ("Syn", "Syn"),
]

PhiID_atoms_full2abbr = {
    full: abbr for full, abbr in zip(PhiID_atoms_full, PhiID_atoms_abbr)
}

PhiID_atoms_abbr2full = {
    abbr: full for full, abbr in zip(PhiID_atoms_full, PhiID_atoms_abbr)
}

def _binarize(v):
    return (v > np.mean(v)).astype(int)