"""
Title	: grid_average.py
To Run	: from grid_average import grid_average
Author	: Frank!
Date	: 20250211
Modf	: n/a

"""
# Import modules
# =======================================
import numpy as np
from scipy.stats import binned_statistic_2d


# Functions and classes
# =======================================

def grid_average(x=None,
                 y=None,
                 values=None,
                 x_bins=100,
                 y_bins=100,
                 statistic='mean'):
    ##############
    # Compute gridded averages of a 1d variable on 2x 1d irregular grids (i.e., lon and lat).
    #
    # Parameters
    # ---------
    # x: ndarray; 1d array of the first dimension (e.g., longitude)
    # y: ndarray; 1d array of the second dimension (e.g., latitude)
    # values: ndarray: 1d variable values for each x and y
    # x_bins: integer or ndarray; number of x bins or bin edges
    # y_bins: integer or ndarray; number of y bins or bin edges
    # statistic: string; can be 'mean', 'std', 'median', 'count', 'sum', 'min', or 'max'
    #
    # Returns
    # -------
    # latitude_corners, longitude_corners : ndarrays; latitude and longitude corners for each pixel
    #                               [len(l1b.granule), 45, 30, 9, 4]
    ##############
    
    # Define bin edges based on min/max values if integers are provided
    x_edges = np.linspace(min(x), max(x), x_bins + 1) if isinstance(x_bins, int) else x_bins
    y_edges = np.linspace(min(y), max(y), y_bins + 1) if isinstance(y_bins, int) else y_bins

    # Compute the binned statistic
    result, _, _, _ = binned_statistic_2d(x,
                                          y,
                                          values,
                                          statistic=statistic,
                                          bins=[x_edges, y_edges])

    return result, x_edges, y_edges
