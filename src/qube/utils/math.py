import numpy as np


def nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.
    
    :param dims: dimensions tuple 
    :return: nans matrix 
    """
    return np.nan * np.ones(dims)