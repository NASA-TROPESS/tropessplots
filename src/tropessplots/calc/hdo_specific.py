"""
Title        :  hdo_specific.py
What is it   :  Routines specific to the MUSES L2_Products_Lite HDO data
Includes     :  split_hdo_h2o_level_value()
                get_all_quadrants_hdo()
                x_species()
                xa_species()
                averaging_kernel()
                observation_error()
                x_test()
Author        : Frank Werner
Date          : 20250905
Modf          : 20xxxxxx: NaN

                               
"""
# Import modules
# =======================================
import numpy as np


# Functions
# =======================================
missing_value=9.969209968386869e+36


def split_hdo_h2o_level_value(hdo_h2o_array):
    return np.split(np.array(hdo_h2o_array), 2, axis=1)


def get_all_quadrants_hdo(array_3d):
    '''
    Y   0
    +---+---+
    | 2 | 1 |
  0 +---+---+ 0
    | 3 | 4 |
    +---+---+ X
        0
    '''
    first_quad, second_quad, third_quad, fourth_quad = [],[],[],[]
    for matrix in array_3d:
        y_split_left_half, y_split_right_half = np.split(np.array(matrix), 2, axis=1)

        first, fourth = np.split(y_split_right_half, 2, axis=0)
        second, third = np.split(y_split_left_half, 2, axis=0)

        first_quad.append(first)
        second_quad.append(second)
        third_quad.append(third)
        fourth_quad.append(fourth)

    return np.array(first_quad), np.array(second_quad), np.array(third_quad), np.array(fourth_quad)


def x_species(x):
    x_dd, x_hh = split_hdo_h2o_level_value(x)
    
    x_ratio = np.divide(x_dd, x_hh, out=np.zeros(x_hh.shape, dtype=float), where=(x_hh != 0))
    
    filler_x = np.where(x_dd < 0)
    x_ratio[filler_x] = missing_value
    
    return x_ratio


def xa_species(xa):
    xa_dd, xa_hh = split_hdo_h2o_level_value(xa)

    xa_ratio = np.divide(xa_dd, xa_hh, out=np.zeros(xa_hh.shape, dtype=float), where=(xa_hh != 0))

    filler_xa = np.where(xa_dd < 0)
    xa_ratio[filler_xa] = missing_value
    xa_dd[filler_xa] = missing_value
    xa_hh[filler_xa] = missing_value
    
    return xa_ratio, xa_dd, xa_hh


def averaging_kernel(ak):
    ak_dh, ak_dd, ak_hd, ak_hh = get_all_quadrants_hdo(ak)

    # set fill value to 0, otherwise it affects the calculation since fill values are getting added together
    ak_dd = np.where(ak_dd != missing_value, ak_dd, 0)
    ak_hd = np.where(ak_hd != missing_value, ak_hd, 0)

    # HDO / H2O ratio averaging kernel
    averaging_kernel_ratio = ak_dd - ak_hd

    # replace 0s back with fill value
    averaging_kernel_ratio = np.where(averaging_kernel_ratio != 0, averaging_kernel_ratio, missing_value)

    return averaging_kernel_ratio


def observation_error(observation_error_covariance):
    obs_dh, obs_dd, obs_hd, obs_hh = get_all_quadrants_hdo(observation_error_covariance)

    # set fill value to 0, otherwise it affects the calculation since fill values are getting added together
    obs_dd = np.where(obs_dd != missing_value, obs_dd, 0)
    obs_hh = np.where(obs_hh != missing_value, obs_hh, 0)
    obs_hd = np.where(obs_hd != missing_value, obs_hd, 0)
    obs_dh = np.where(obs_dh != missing_value, obs_dh, 0)

    # HDO / H2O ratio observation error covariance
    observation_error_ratio = obs_dd + obs_hh - obs_hd - obs_dh

    # replace 0s back with fill value
    observation_error_ratio = np.where(observation_error_ratio != 0, observation_error_ratio, missing_value)

    return observation_error_ratio


def x_test(pressure=None,
           averaging_kernel_ratio=None,
           x_ratio=None,
           xa_ratio=None,
          ):
    
    # x_test calculations
    p_dum = pressure[0]                         # pdum = yar.pressure(0:num-1,good)
    uu = np.where(p_dum > 0)[0]             # uu = where(pdum gt 0)

    ak_dum = averaging_kernel_ratio[0]
    ak_dum = ak_dum[uu, :][:, uu]           # akdum = ak_r(*,*,0), akdum(uu,uu,*)
    xa_dum = xa_ratio[0][uu]                # xadum = xa_r(*,0), xadum(uu)

    # take the first observation as x_true
    x_true = x_ratio[0][uu]                 # x_true = x_r(*,0), x_true(uu)

    # x_est(uu) = exp( alog(xadum(uu)) + akdum(uu,uu,*) ## ( alog( x_true(uu)) - alog(xadum(uu)) ) )
    x_test_est_log = np.log(xa_dum) + ak_dum @ (np.log(x_true) - np.log(xa_dum))
    x_test = np.exp(x_test_est_log)

    return x_test