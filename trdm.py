import numpy as np
from kmodes.kmodes import KModes
from numpy import exp, log
from scipy.misc import logsumexp


class TensorRankDecompositionModel:
    def __init__(self,
                 shape,
                 n_components,
                 tol=1e-3,
                 max_iter=100,
                 verbose=0,
                 verbose_interval=10):
        self.ds = shape
        self.k = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.epsilon = 1e-7

        self.w = np.ones([self.k]) / self.k
        self.us = [log(np.random.rand(self.k, d)) for d in self.ds]
        for u in self.us:
            u -= logsumexp(u, axis=1)[:, None]

    def init_km(self, data):
        km = KModes(
            n_clusters=self.k, max_iter=self.max_iter, verbose=self.verbose)
        category = km.fit_predict(data)
        self.w = np.bincount(category, minlength=self.k)
        # Add-one regularization
        self.w = (self.w + 1 / self.k) / (data.shape[0] + 1)

        for kk in range(self.k):
            data_k = data[category == kk, :]
            if data_k.size > 0:
                for i, u in enumerate(self.us):
                    u[kk, :] = np.bincount(data_k[:, i], minlength=self.ds[i])
                    u[kk, :] = (u[kk, :] + 1 / self.ds[i]) / (
                        data_k.shape[0] + 1)
        for u in self.us:
            u[:] = log(u)

    def log_pnk_l(self, data):
        log_pnk = [u[:, data[:, i]].T for i, u in enumerate(self.us)]
        log_pnk = np.sum(log_pnk, axis=0)
        log_p = log((self.w * exp(log_pnk)).sum(axis=1))
        log_l = log_p.sum()
        return log_pnk, log_l

    def fit(self, data):
        assert isinstance(data, np.ndarray)
        assert np.issubdtype(data.dtype, np.integer)
        assert len(data.shape) == 2
        assert data.shape[1] == len(self.ds)
        assert (data >= 0).all() and (data < self.ds).all()

        log_l_old = np.inf
        for iter in range(self.max_iter):
            log_pnk, log_l = self.log_pnk_l(data)
            log_l_diff = log_l - log_l_old
            if abs(log_l_diff) < self.tol:
                break
            log_l_old = log_l

            if self.verbose >= 1 and iter % self.verbose_interval == 0:
                print(iter, log_l, log_l_diff)

            log_wk = logsumexp(log_pnk, axis=0)
            wk = exp(log_wk)
            self.w = wk / wk.sum()

            for kk in range(self.k):
                for i, u in enumerate(self.us):
                    u[kk, :] = np.bincount(
                        data[:, i],
                        weights=exp(log_pnk[:, kk]),
                        minlength=self.ds[i])
            for u in self.us:
                u[:] = log(u + self.epsilon)
                u -= logsumexp(u, axis=1)[:, None]

    def sample(self, n_samples=1):
        category = np.random.choice(self.k, size=n_samples, p=self.w)
        out = [[
            np.random.choice(d, p=exp(u[kk, :]))
            for d, u in zip(self.ds, self.us)
        ] for kk in category]
        out = np.array(out, dtype=np.int64)
        return out, category

    def score(self, data):
        _, log_l = self.log_pnk_l(data)
        out = log_l / data.shape[0]
        return out
