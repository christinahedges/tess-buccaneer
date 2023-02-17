"""Utility functions to make design matrices"""
import numpy as np
import numpy.typing as npt
from scipy import sparse


def vstack(dm: npt.ArrayLike, n: int) -> sparse.csr_matrix:
    """Creates an independent vertical stack of `n` versions of `dm`"""
    npoints = dm.shape[0] * n
    ncomps = dm.shape[1] * n
    X = sparse.lil_matrix((npoints, ncomps))
    idx, jdx = 0, 0
    for count in range(n):
        X[idx : idx + dm.shape[0], jdx : jdx + dm.shape[1]] = dm
        idx = idx + dm.shape[0]
        jdx = jdx + dm.shape[1]
    return X


def hstack(dm: npt.ArrayLike, n: int, X=None, offset: int = 0) -> sparse.csr_matrix:
    """Creates an independent horizontal stack of `n` versions of `dm`"""
    npoints = dm.shape[0] * n
    ncomps = dm.shape[1] * n
    if X is None:
        X = sparse.lil_matrix((npoints, ncomps))
    idx, jdx = 0, 0
    for count in range(n):
        if not (((count + offset) < 0) | ((count + offset) >= n)):
            X[count + offset :: n, jdx : jdx + dm.shape[1]] += dm
        idx = idx + dm.shape[0]
        jdx = jdx + dm.shape[1]
    return X
