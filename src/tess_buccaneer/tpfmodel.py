"""Classes to model TPFs"""

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.stats import sigma_clip
from fbpca import pca
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse

from . import PACKAGEDIR
from .utils import hstack, vstack

_straplocs = np.asarray(pd.read_csv(f"{PACKAGEDIR}/data/strap_locs.csv", header=None))[
    :, 0
]


def enlarge_mask(mask):
    a, b = mask[1:], mask[:-1]
    mask[:-1] |= a
    mask[1:] |= b


def get_tpf_tmasks(tpf):
    bs = np.asarray([0, *np.where(np.diff(tpf.time.value) > 0.2)[0], tpf.shape[0]])
    return np.asarray(
        [
            np.in1d(np.arange(tpf.shape[0]), np.arange(bs[idx], bs[idx + 1]))
            for idx in range(len(bs) - 1)
        ]
    )


def clean_tpf(tpf, pix_mask):
    masks = get_tpf_tmasks(tpf)
    bkg_mask = []
    for mask in masks:
        y = tpf.flux.value[mask][:, pix_mask].mean(axis=(1))
        bad_bkg = (y > 500) | (np.abs(np.gradient(y)) > 15)
        _ = [
            enlarge_mask(bad_bkg)
            for count in range(int(0.25 // np.median(np.diff(tpf.time.value))))
        ]
        bkg_mask.append(bad_bkg)
    bkg_mask = np.hstack(bkg_mask)
    return tpf[~bkg_mask]


class TPFModel(object):
    """Class to model a TPF"""

    def __init__(self, tpf: lk.targetpixelfile.TargetPixelFile, polyorder=1):
        self.include_ts_model = False
        m = np.percentile(tpf.flux.value, 10, axis=0)
        self.star_mask = ~larger_aper(sigma_clip(m, sigma=3).mask).ravel()
        self.tpf = clean_tpf(tpf, self.star_mask.reshape(tpf.shape[1:]))

        if (self.tpf.flux.value > 1e5).any():
            raise ValueError("Contains saturated pixels, I cant cope with that yet.")
        self.R, self.C = np.mgrid[: self.tpf.shape[1], : self.tpf.shape[2]].astype(
            float
        )
        self.strapmask = np.in1d(
            self.C[0].astype(int) + self.tpf.column, _straplocs + 44
        )
        self.R -= self.tpf.shape[1] / 2
        self.C -= self.tpf.shape[2] / 2
        self.t = (self.tpf.time.value - self.tpf.time.value.mean()) / (
            self.tpf.time.value.max() - self.tpf.time.value.min()
        )
        self.time_masks = get_tpf_tmasks(self.tpf)

        X1 = np.vstack(
            [
                self.R.ravel() ** idx * self.C.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T
        X2 = np.vstack(
            [
                (self.R.ravel() ** idx)
                * (self.strapmask * np.ones(self.C.shape)).ravel()
                for idx in range(1)
            ]
        ).T
        self.X = np.hstack([X1, X2])

        self.meds, self.g1s, self.g2s = [], [], []
        for mask in self.time_masks:
            m = np.median(self.tpf.flux.value[mask], axis=0)

            # The bkg zeropoint
            bkg0 = self.X.dot(
                np.linalg.solve(
                    self.X[self.star_mask].T.dot(self.X[self.star_mask]),
                    self.X[self.star_mask].T.dot(m.ravel()[self.star_mask]),
                )
            ).reshape(m.shape)
            self.meds.append(m - bkg0)
            self.g1s.append(np.gradient(self.meds[-1])[0])
            self.g2s.append(np.gradient(self.meds[-1])[1])

        # self.minframe = m - self.bkg0
        self.minframe = np.mean(self.meds, axis=0)
        self.use_pca_bkg = False
#        self._get_PCA_bkg_model()
        self.S = self.X.copy()
        self._get_sparse_arrays()

    @property
    def shape(self):
        return self.tpf.shape

    def _get_PCA_bkg_model(self):
        """This gets the components for the PCA bkg model"""
        self.use_pca_bkg = True
        resids = self.tpf.flux.value - self.minframe

        # Get the PCA components for the background data, up to 10
        self.U, self.s, self.V = pca(
            resids[:, self.star_mask.reshape(self.shape[1:])],
            np.min([np.max([1, (self.star_mask).sum() // 100]), 7]),
        )

        # Use a polynomial model to in-fill the "V" component wherever there are sources!!
        self.V_weights = np.linalg.solve(
            self.X[self.star_mask].T.dot(self.X[self.star_mask]),
            self.X[self.star_mask].T.dot(self.V.T),
        )
        self.V_model = self.X.dot(self.V_weights).T

        # Find only the components that are well modelled by a simple polynomial
        S, self.comps = [np.ones(np.product(self.shape[1:]))], []
        for vdx in range(self.V.shape[0]):
            chi1 = (self.V[vdx] - self.V_model[vdx][self.star_mask]) ** 2 / np.abs(
                self.V[vdx] + self.V[vdx].min() + 1
            )
            chi2 = (self.V[vdx] - np.median(self.V[vdx])) ** 2 / np.abs(
                self.V[vdx] + self.V[vdx].min() + 1
            )

            if chi1.sum() / chi2.sum() < 0.6:
                self.comps.append(vdx)
                S.append(self.X.dot(self.V_weights[:, vdx]))
        self.pca_model = (
            self.U[:, np.asarray(self.comps)]
            .dot(np.diag(self.s[self.comps]))
            .dot(self.X.dot(self.V_weights[:, self.comps]).T)
            .reshape(self.shape)
        )
        self.S = np.asarray(S).T

    def _get_sparse_arrays(self):
        resids = self.tpf.flux.value - self.minframe
        model = self.S.dot(
            np.linalg.solve(
                self.S[self.star_mask].T.dot(self.S[self.star_mask]),
                self.S[self.star_mask].T.dot(
                    resids[:, self.star_mask.reshape(self.shape[1:])].T
                ),
            )
        ).T
        model = model.reshape(self.shape)
        # medframe = np.median(self.tpf.flux.value - model, axis=0)
        # gframe = np.gradient(medframe)
        mask3d = (
            self.star_mask.reshape(self.shape[1:])[None, :, :]
            * np.ones(self.shape, bool)
        ).ravel()

        sM1 = vstack(self.S, self.shape[0]).tocsr()
        sM2 = []
        for mdx, mask in enumerate(self.time_masks):
            m, g1, g2, t = [
                sparse.lil_matrix(
                    (
                        self.meds[mdx][None, :, :] * mask.astype(float)[:, None, None]
                    ).ravel()
                ),
                sparse.lil_matrix(
                    (
                        self.g1s[mdx][None, :, :] * mask.astype(float)[:, None, None]
                    ).ravel()
                ),
                sparse.lil_matrix(
                    (
                        self.g2s[mdx][None, :, :] * mask.astype(float)[:, None, None]
                    ).ravel()
                ),
                sparse.lil_matrix(
                    (
                        self.t[:, None, None]
                        * np.ones(self.shape)
                        * mask.astype(float)[:, None, None]
                    ).ravel()
                ),
            ]
            sM2.append(
                sparse.vstack(
                    [
                        m,
                        g1,
                        g2,
                        g1.multiply(g2),
                        g1.multiply(t),
                        g2.multiply(t),
                        g1.multiply(g2).multiply(t),
                        g1.multiply(t).multiply(t),
                        g2.multiply(t).multiply(t),
                        g1.multiply(g2).multiply(t).multiply(t),
                    ])
                .T
            )
        sM2 = sparse.hstack(sM2)
        self.sM = sparse.hstack([sM1, sM2]).tocsr()
        self.prior_mu = np.zeros(self.sM.shape[1])
        self.prior_sigma = np.ones(self.sM.shape[1]) * np.inf
        sm1_weights = np.linalg.solve(
            sM1[mask3d].T.dot(sM1[mask3d]).toarray(),
            sM1[mask3d].T.dot(self.tpf.flux.value.ravel()[mask3d]),
        )
        self.prior_mu[: sM1.shape[1]] = sm1_weights
        self.prior_sigma[: sM1.shape[1]] = 10
        sm2_weights = np.linalg.solve(
            self.sM.T.dot(self.sM).toarray(), self.sM.T.dot(self.tpf.flux.value.ravel())
        )
        self.prior_mu[-sM2.shape[1] :] = sm2_weights[-sM2.shape[1] :]
        self.prior_sigma[-sM2.shape[1] :] = 1

        self.sM1_idx = np.arange(sM1.shape[1])
        self.sM2_idx = np.arange(sM1.shape[1], sM1.shape[1] + sM2.shape[1])

        self.sigma_c_inv = sparse.diags(1 / self.prior_sigma**2, format="csr")

    def include_model(self, TSModel, prior_sigma=1):
        self.include_ts_model = True
        self.ts_model = TSModel
        sM3 = TSModel.X(self.tpf.time.value, self.shape[1:])
        self.sM3_idx = np.arange(
            len(self.sM1_idx) + len(self.sM2_idx),
            len(self.sM1_idx) + len(self.sM2_idx) + sM3.shape[1],
        )

        self.prior_mu = np.hstack([self.prior_mu, np.zeros(sM3.shape[1])])
        self.prior_sigma = np.hstack(
            [self.prior_sigma, prior_sigma * np.ones(sM3.shape[1])]
        )

        sigma_c_inv = sparse.lil_matrix(
            (
                len(self.sM1_idx) + len(self.sM2_idx) + len(self.sM3_idx),
                len(self.sM1_idx) + len(self.sM2_idx) + len(self.sM3_idx),
            )
        )
        sigma_c_inv[
            : self.sigma_c_inv.shape[0], : self.sigma_c_inv.shape[1]
        ] = self.sigma_c_inv.copy()
        cov = TSModel.estimate_covariance_matrix(
            prior_mu=0, prior_sigma=prior_sigma, shape=self.shape[1:], nsamples=2000
        )
        covi = np.linalg.inv(cov)
        #        covi = sparse.diags(1/(np.ones(sM3.shape[1]) * prior_sigma)**2, format='csr').toarray()
        sigma_c_inv[-covi.shape[0] :, -covi.shape[1] :] = covi
        self.sigma_c_inv = sigma_c_inv.tocsr()
        self.sM = sparse.hstack([self.sM, sM3]).tocsr()

    def fit(self):
        good_pixel_mask = np.ones(np.product(self.shape), bool)
        for count in range(2):
            sigma_w_inv = (
                self.sM[good_pixel_mask].T.dot(self.sM[good_pixel_mask]).toarray()
                + self.sigma_c_inv
            )
            self.sm_weights = np.linalg.solve(
                sigma_w_inv,
                self.sM[good_pixel_mask].T.dot(
                    self.tpf.flux.value.ravel()[good_pixel_mask]
                )
                + (self.prior_mu * self.sigma_c_inv.diagonal()),
            )

            smodel = self.sM.dot(self.sm_weights).reshape(self.shape)
            resids = ((self.tpf.flux.value - smodel) / self.tpf.flux_err.value).ravel()
            good_pixel_mask = ~sigma_clip(
                resids, sigma=6, cenfunc=lambda x, axis: x
            ).mask

        return smodel

    @property
    def bkg_model(self):
        if not hasattr(self, "sm_weights"):
            raise ValueError("No weights exist, run `fit` method.")
        return (
            self.sM[:, self.sM1_idx]
            .dot(self.sm_weights[self.sM1_idx])
            .reshape(self.shape)
        )

    @property
    def star_model(self):
        if not hasattr(self, "sm_weights"):
            raise ValueError("No weights exist, run `fit` method.")
        return (
            self.sM[:, self.sM2_idx]
            .dot(self.sm_weights[self.sM2_idx])
            .reshape(self.shape)
        )

    def plot_PCA_model(self):
        if not self.use_pca_bkg:
            return None
        fig, ax = plt.subplots(self.V.shape[0], 2, figsize=(6, 2 * self.V.shape[0]))
        for vdx in range(self.V.shape[0]):
            if vdx in self.comps:
                cmap = "viridis"
            else:
                cmap = "Greys"
            l1, l2 = np.ones((2, *self.shape[1:])) * np.nan
            l1[self.star_mask.reshape(self.shape[1:])] = self.V[vdx]
            l2 = self.V_model[vdx].reshape(self.shape[1:])
            vmin, vmax = np.sort(np.nanpercentile(l1, (5, 95)))
            ax[vdx, 0].imshow(l1, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[vdx, 1].imshow(l2, vmin=vmin, vmax=vmax, cmap=cmap)
            if vdx == 0:
                ax[vdx, 0].set(title=f"PCA `V`")
                ax[vdx, 1].set(title=f"Infilled PCA `V`")
            ax[vdx, 0].set(ylabel=f"Component {vdx}", xticklabels=[], yticklabels=[])
            ax[vdx, 1].set(xticklabels=[], yticklabels=[])
        plt.tight_layout()
        return fig

    def plot_starmask(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        self.tpf.plot(ax=ax[0])
        ax[1].imshow(self.star_mask.reshape(self.shape[1:]), cmap="Greys")
        ax[1].set(
            ylabel=f"Pixel Row",
            xlabel="Pixel Column",
            title="Pixel Mask",
            xticklabels=[],
            yticklabels=[],
        )
        plt.tight_layout()
        return fig

    def plot_medframes(self):
        fig, ax = plt.subplots(3, np.max([len(self.meds), 2]), figsize=(2.5 * np.max([len(self.meds), 2]), 8))
        
        for jdx, ar, label in zip(
            [0, 1, 2],
            [self.meds, self.g1s, self.g2s],
            ["Median Frame", "Gradient 1", "Gradient 2"],
        ):
            vmin, vmax = np.percentile(np.asarray(ar), (10, 90))
            for idx, v in enumerate(ar):
                ax[jdx, idx].imshow(v, cmap="viridis", vmin=vmin, vmax=vmax)
                ax[jdx, idx].set(
                    ylabel=f"Pixel Row",
                    xlabel="Pixel Column",
                    title=label,
                    xticklabels=[],
                    yticklabels=[],
                )
        plt.tight_layout()
        return fig

    def plot_resids(self, frame, vmin=None, vmax=None, cmap="coolwarm"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        resids1 = self.tpf.flux.value[frame] - np.median(self.tpf.flux.value, axis=0)
        resids1 -= np.median(resids1[self.star_mask.reshape(self.shape[1:])], axis=0)
        resids2 = (
            self.tpf.flux.value[frame] - self.bkg_model[frame] - self.star_model[frame]
        )
        if vmin is None:
            vmin = np.min([np.percentile(resids1, 3), np.percentile(resids2, 3)])
        if vmax is None:
            vmax = np.max([np.percentile(resids1, 97), np.percentile(resids2, 97)])
        im1 = ax[0].imshow(resids1, vmin=vmin, vmax=vmax, cmap=cmap)
        im2 = ax[1].imshow(resids2, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im1, ax=ax[0])
        plt.colorbar(im2, ax=ax[1])
        ax[0].set(title=f"Frame {frame}\nSimple Mean Subtraction Residuals")
        ax[1].set(title=f"Frame {frame}\nFull Model Residuals")
        return fig

    def plot_ts_model(self):
        if self.include_ts_model:
            L = self.sm_weights[self.sM3_idx].reshape(self.shape[1:])
            fig = plt.figure(figsize=(10, 3))
            ax = plt.subplot2grid((1, 4), (0, 3))
            ax.set(xlabel="Pixel Column", ylabel="Pixel Row", title="Time Series Power")
            im = ax.imshow(L, vmin=-1, vmax=2)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Component Power [$e^-s^{-1}$]")
            ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
            ax.plot(self.tpf.time.value, self.ts_model.model(self.tpf.time.value))
            ax.set(title="Time Series Model", xlabel="Time", ylabel="Value")
            return fig

    def make_pdf(self, filename: str = "buccaneer_output.pdf"):
        p = PdfPages(filename)
        figs = [
            self.plot_starmask(),
            self.plot_medframes(),
            *[
                self.plot_resids(
                    frame=(idx) * (self.shape[0] - 1) // 4, vmin=-2, vmax=2
                )
                for idx in np.linspace(0, 4, 5, dtype=int)
            ],
        ]
        if self.include_ts_model:
            figs.append(self.plot_ts_model())
        if self.use_pca_bkg:
            figs.append(self.plot_PCA_model())

        for fig in figs:
            fig.savefig(p, format="pdf")

        p.close()
        plt.close("all")


def larger_aper(aper: np.ndarray) -> np.ndarray:
    """Enlarges an aperture by one pixel"""
    laper = aper | (np.hypot(*np.gradient(np.asarray(aper, float))) != 0)
    return laper
