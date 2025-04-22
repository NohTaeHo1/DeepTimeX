import numpy as np

def jitter(X, sigma=0.05):
    return X + np.random.normal(loc=0.0, scale=sigma, size=X.shape)


def scaling(X, sigma=0.1):
    scaling_factors = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1))
    return X * scaling_factors
