import os

import h5py
import numpy as np
from scipy.special import ndtri
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

in_dir = 'data_8300'
out_dir = 'data'


def n_parameters(model):
    n_features = model.means_.shape[1]
    cov_params = model.n_components * n_features * (n_features + 1) // 2
    mean_params = n_features * model.n_components
    total_params = cov_params + mean_params + model.n_components - 1
    return total_params


def bic(model, data):
    return (-2 * model.score(data) * data.shape[0] +
            n_parameters(model) * np.log(data.shape[0]))


def get_mean_cov(data, use_cdf=False, epsilon=1e-7):
    if use_cdf:
        data = np.clip(data, epsilon, 1 - epsilon)
        data = ndtri(data)
    data = data.T
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    return mean, cov


# Choose n_components to minimize bic
def get_gmm(data, n_components, bayesian=False, use_cdf=False, epsilon=1e-7):
    if use_cdf:
        data = np.clip(data, epsilon, 1 - epsilon)
        data = ndtri(data)
    if bayesian:
        gmm = BayesianGaussianMixture(
            n_components=n_components, max_iter=1000, verbose=2)
    else:
        gmm = GaussianMixture(
            n_components=n_components, max_iter=1000, verbose=2)
    gmm.fit(data)
    print('weight')
    print(gmm.weights_)
    print('BIC', bic(gmm, data))
    return gmm


if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'body_params.hdf5'), 'r') as f:
        body_params = np.array(f['data'], dtype=np.float64)

    gmm = get_gmm(body_params, 3)

    with h5py.File(os.path.join(out_dir, 'body_params.hdf5'), 'w') as f:
        f.create_dataset('weight', data=gmm.weights_)
        f.create_dataset('mean', data=gmm.means_)
        f.create_dataset('cov', data=gmm.covariances_)
