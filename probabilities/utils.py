import numpy as np

def project_onto_simplex(v, z=1):
    n_features = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def distribution_to_hist(distrib, precision=100):
    return np.random.choice(len(distrib), size=(precision,), p=distrib)
