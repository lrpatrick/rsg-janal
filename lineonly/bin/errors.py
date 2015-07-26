"""
Author: LRP
Date: 30-01-2015
Description:
Define the errors for a set of parameters by repeating

All dependencies are contained within astropy
Code is written to conform with PEP8 style guide

References:
Gazak (2014) Thesis

"""


def err(mspec, ospec):
    """
    Steps:
        1. add random gaussian noise to model -- with size ~observed
        2. Recompute Xi-squared grid
        3. Re-fit parameters
    """
    # Add noise
    nmod = mspec + np.random.normal(0, 0.01, mspec.shape[0])
    # Compute Xi-squared



