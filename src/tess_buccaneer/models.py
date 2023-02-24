"""Models for common astrophysical variability"""
from dataclasses import dataclass

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.timeseries.periodograms import BoxLeastSquares
from scipy import sparse

from .utils import hstack


@dataclass
class BoxTransit:
    """Class that handles box shaped transits"""

    period: u.Quantity
    t0: u.Quantity
    duration: u.Quantity
    eclipse: bool = False
    matrixtype: str = 'pixel'

    @property
    def kernel(self):
        return np.asarray(
            [[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]
        )

    def kernel_locs(self, shape: tuple):
        k = self.kernel.shape[0]
        l = np.mgrid[-1:2, -1:2]
        return l[0] * (shape[1]) + l[1]

    def model(self, time: npt.ArrayLike) -> np.ndarray:
        ts = (
            -BoxLeastSquares([], [])
            .transit_mask(time, self.period, self.duration, self.t0)
            .astype(float)
        )
        if not self.eclipse:
            return ts[:, None]
        else:
            ec = (
                -BoxLeastSquares([], [])
                .transit_mask(time, period, duration, t0 + period / 2)
                .astype(float)
            )
            return np.vstack([ts, ec]).T

    def _X(self, time: npt.ArrayLike, shape: tuple) -> sparse.csr_matrix:
        ts_model = self.model(time)
        X = hstack(ts_model, np.product(shape))
        return X.tocsr()

    def _X_gaussian(self, time: npt.ArrayLike, shape: tuple) -> sparse.csr_matrix:
        ts_model = self.model(time)
        X = None
        kernel = self.kernel.ravel()
        locs = self.kernel_locs(shape).ravel()
        for amp, offset in zip(kernel, locs):
            X = hstack(ts_model * amp, np.product(shape), X=X, offset=offset)
        return X.tocsr()

    def X(self, time: npt.ArrayLike, shape: tuple) -> sparse.csr_matrix:
        if self.matrixtype == 'pixel':
            return self._X(time=time, shape=shape)
        elif self.matrixtype == 'gaussian':
            return self._X_gaussian(time=time, shape=shape)
        else:
            raise ValueError(f"No such type as {self.matrixtype}")

    def Xfootprint(self, shape: tuple) -> sparse.csr_matrix:
        if self.matrixtype == 'pixel':
            X = hstack(np.atleast_2d(1), np.product(shape))
            return X
        elif self.matrixtype == 'gaussian':
            X = None
            kernel = self.kernel.ravel()
            locs = self.kernel_locs(shape).ravel()
            for amp, offset in zip(kernel, locs):
                X = hstack(np.atleast_2d(1) * amp, np.product(shape), X=X, offset=offset)
            return X
        else:
            raise ValueError(f"No such type as {self.matrixtype}")

    def estimate_covariance_matrix(self, prior_mu, prior_sigma, shape, nsamples=2000):
        X = self.Xfootprint(shape)
        w = np.random.normal(prior_mu, prior_sigma, size=(X.shape[1], nsamples))
        draws = X.dot(w)
        return np.cov(draws)


@dataclass
class Flare:
    """Class that handles flares"""

    t0: u.Quantity
    fexp: u.Quantity
    gexp: u.Quantity

    def model(self, time: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Sinusoid:
    """Class that handles periodic/sinusoidal variability"""

    period: u.Quantity
    nterms: int = 1

    def model(self, time: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SuperNova:
    """Class that handles supernova like light curves"""

    t0: u.Quantity
    rise: u.Quantity
    fall: u.Quantity

    def model(self, time: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError
