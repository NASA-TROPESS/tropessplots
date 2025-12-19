"""
Title        :  airs_omi.py
What is it   :  Routines to read and write AIRS_OMI data
Includes     :  read_l2summary()
                read_l2standard()
Author       :  Frank Werner
Date         :  20251008
Modf         :  20251009: Adjusted read_l2standard() to only read certain variables if they exist
                20251021: Changed the row proxy from 10000 to 15000 in read_l2summary() and read_l2standard()

"""
# Import modules
# =======================================
import numpy as np

from tropessplots.shared.read_nc import read_nc
from tropessplots.shared.date_to_julian_day import date_to_julian_day


# Functions and classes
# =======================================
class _ReadL2Summary ():
    ##############
    # This creates a class for the TROPESS L2 Summary data.
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # self : an empty object/class; an object with the Summary TROPESS Products
    ##############
    def __init__(self,
                 number_of_files,
                 global_attrs,
                 date,
                 airs_granule,
                 airs_along_index,
                 airs_cross_index,
                 airs_view_ang,
                 omi_granule,
                 omi_along_index,
                 omi_cross_index_uv1,
                 omi_cross_index_uv2,
                 omi_view_ang,
                 latitude,
                 longitude,
                 pressure,
                 utc,
                 time,
                 land_flag,
                 day_night_flag,
                 target_id,
                 doy,
                 col,
                 col_t,
                 col_ut,
                 col_error,
                 col_dry_air,
                 x,
                 x_prior,
                 species):

        self.number_of_files = number_of_files
        self.global_attrs = global_attrs

        self.date = date
        self.airs_granule = airs_granule
        self.airs_along_index = airs_along_index
        self.airs_cross_index = airs_cross_index
        self.airs_view_ang = airs_view_ang
        self.omi_granule = omi_granule
        self.omi_along_index = omi_along_index
        self.omi_cross_index_uv1 = omi_cross_index_uv1
        self.omi_cross_index_uv2 = omi_cross_index_uv2
        self.omi_view_ang = omi_view_ang
        self.latitude = latitude
        self.longitude = longitude
        self.pressure = pressure
        self.utc = utc
        self.time = time
        self.land_flag = land_flag
        self.day_night_flag = day_night_flag
        self.target_id = target_id
        self.doy = doy

        self.col = col
        self.col_t = col_t
        self.col_ut = col_ut
        self.col_error = col_error
        self.col_dry_air = col_dry_air
        self.x = x
        self.x_prior = x_prior

        self.species = species


def read_l2summary(files=None,
                   verbose=0):
    ##############
    # Read (multiple) TROPESS L2 Summary file(s).
    #
    # Parameters
    # ---------
    # file: string; full path to a CrIS L2 Summary file
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # _ReadL2Summary : an object/class; an object with the CrIS TROPESS Summary data
    ##############

    # Define species.
    species = "O3"

    # Define datasets
    data_sets = ['/longitude',
                 '/latitude',
                 '/time',
                 '/datetime_utc',
                 '/year_fraction',
                 '/altitude',
                 '/pressure',
                 '/target_id',
                 '/geolocation/airs_granule',
                 '/geolocation/airs_atrack',
                 '/geolocation/airs_xtrack',
                 '/geolocation/airs_view_ang',
                 '/geolocation/omi_granule',
                 '/geolocation/omi_atrack',
                 '/geolocation/omi_xtrack_uv1',
                 '/geolocation/omi_xtrack_uv2',
                 '/geolocation/omi_view_ang',
                 '/geophysical/land_flag',
                 '/geophysical/day_night_flag',
                 '/x',
                 '/xa',
                 '/col',
                 '/col_error',
                 '/col_dry_air',
                 '/col_t',
                 '/col_ut']
    n_col = 26

    # Define arrays
    number_of_files = len(files)
    global_attrs = {}

    n_rows = 15000*len(files)

    date = np.zeros((n_rows), dtype=int)
    airs_granule = np.zeros((n_rows), dtype=np.int32)
    airs_along_index = np.zeros((n_rows), dtype=np.int32)
    airs_cross_index = np.zeros((n_rows), dtype=np.int32)
    airs_view_ang = np.zeros((n_rows), dtype=np.float32)
    omi_granule = np.zeros((n_rows), dtype=np.int32)
    omi_along_index = np.zeros((n_rows), dtype=np.int32)
    omi_cross_index_uv1 = np.zeros((n_rows), dtype=np.int32)
    omi_cross_index_uv2 = np.zeros((n_rows), dtype=np.int32)
    omi_view_ang = np.zeros((n_rows), dtype=np.float32)
    latitude = np.zeros((n_rows), dtype=np.float32)
    longitude = np.zeros((n_rows), dtype=np.float32)
    pressure = np.zeros((n_rows, n_col), dtype=np.float32)
    utc = np.zeros((n_rows), dtype=np.float32)
    time = np.zeros((n_rows), dtype=np.float32)
    land_flag = np.zeros((n_rows), dtype=np.int32)
    day_night_flag = np.zeros((n_rows), dtype=np.int32)
    target_id = np.zeros((n_rows), dtype=np.int32)
    doy = np.zeros((n_rows), dtype=np.int32)

    # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19], 'for_dobson_units_multiply_by': array([2241.1475], dtype=float32)
    col = np.zeros((n_rows), dtype=np.float32)
    # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19], 'for_dobson_units_multiply_by': array([2241.1475], dtype=float32)
    col_t = np.zeros((n_rows), dtype=np.float32)
    # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19], 'for_dobson_units_multiply_by': array([2241.1475], dtype=float32)
    col_ut = np.zeros((n_rows), dtype=np.float32)
    # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19]
    col_error = np.zeros((n_rows), dtype=np.float32)
    # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19]
    col_dry_air = np.zeros((n_rows), dtype=np.float32)
    x = np.zeros((n_rows, n_col), dtype=np.float32)  # vmr in ppbv
    x_prior = np.zeros((n_rows, n_col), dtype=np.float32)  # vmr in ppbv

    count = 0
    for i_files in range(0, number_of_files):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=data_sets)

        # Fill global attributes
        global_attrs[i_files] = data.global_attrs

        # Find length of the variables in this file
        l = len(data.values['/geolocation/airs_granule'][:])

        # Fill variables
        pos_date = files[i_files].find(
            'TROPESS_AIRS-Aqua_OMI-Aura_L2_Summary_')
        pos_date2 = files[i_files].find('_', pos_date+40, -1)
        date[count:count+l] = int(files[i_files][pos_date2+1:pos_date2+5] +
                                  files[i_files][pos_date2+5:pos_date2+7] +
                                  files[i_files][pos_date2+7:pos_date2+9])
        airs_granule[count:count +
                     l] = data.values['/geolocation/airs_granule'][:]
        airs_along_index[count:count +
                         l] = data.values['/geolocation/airs_atrack'][:]
        airs_cross_index[count:count +
                         l] = data.values['/geolocation/airs_xtrack'][:]
        airs_view_ang[count:count +
                      l] = data.values['/geolocation/airs_view_ang'][:]
        omi_granule[count:count+l] = data.values['/geolocation/omi_granule'][:]
        omi_along_index[count:count +
                        l] = data.values['/geolocation/omi_atrack'][:]
        omi_cross_index_uv1[count:count +
                            l] = data.values['/geolocation/omi_xtrack_uv1'][:]
        omi_cross_index_uv2[count:count +
                            l] = data.values['/geolocation/omi_xtrack_uv2'][:]
        omi_view_ang[count:count +
                     l] = data.values['/geolocation/omi_view_ang'][:]
        latitude[count:count+l] = data.values['/latitude'][:]
        longitude[count:count+l] = data.values['/longitude'][:]
        pressure[count:count+l] = data.values['/pressure'][:]
        dummy_utc = data.values['/datetime_utc'][:, 3] + \
            data.values['/datetime_utc'][:, 4]/60 + \
            data.values['/datetime_utc'][:, 5]/3600
        utc[count:count+l] = dummy_utc
        time[count:count+l] = data.values['/time'][:]
        land_flag[count:count +
                  l] = data.values['/geophysical/land_flag'][:]
        day_night_flag[count:count +
                       l] = data.values['/geophysical/day_night_flag'][:]
        target_id[count:count+l] = data.values['/target_id'][:]
        doy[count:count+l] = date_to_julian_day(int(files[i_files][pos_date2+1:pos_date2+5]),
                                                int(files[i_files]
                                                    [pos_date2+5:pos_date2+7]),
                                                int(files[i_files][pos_date2+7:pos_date2+9]))
        col[count:count+l] = data.values['/col'][:]
        col_t[count:count+l] = data.values['/col_t'][:]
        col_ut[count:count+l] = data.values['/col_ut'][:]
        col_error[count:count+l] = data.values['/col_error'][:]
        col_dry_air[count:count+l] = data.values['/col_dry_air'][:]
        x[count:count+l, :] = data.values['/x'][:, :]
        x_prior[count:count+l, :] = data.values['/xa'][:]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    airs_granule = airs_granule[0:count]
    airs_along_index = airs_along_index[0:count]
    airs_cross_index = airs_cross_index[0:count]
    airs_view_ang = airs_view_ang[0:count]
    omi_granule = omi_granule[0:count]
    omi_along_index = omi_along_index[0:count]
    omi_cross_index_uv1 = omi_cross_index_uv1[0:count]
    omi_cross_index_uv2 = omi_cross_index_uv2[0:count]
    omi_view_ang = omi_view_ang[0:count]
    latitude = latitude[0:count]
    longitude = longitude[0:count]
    pressure = pressure[0:count]
    utc = utc[0:count]
    time = time[0:count]
    land_flag = land_flag[0:count]
    day_night_flag = day_night_flag[0:count]
    target_id = target_id[0:count]
    doy = doy[0:count]

    col = col[0:count]
    col_t = col_t[0:count]
    col_ut = col_ut[0:count]
    col_error = col_error[0:count]
    col_dry_air = col_dry_air[0:count]
    x = x[0:count]
    x_prior = x_prior[0:count]

    # Return data
    return(_ReadL2Summary(number_of_files,
                          global_attrs,
                          date,
                          airs_granule,
                          airs_along_index,
                          airs_cross_index,
                          airs_view_ang,
                          omi_granule,
                          omi_along_index,
                          omi_cross_index_uv1,
                          omi_cross_index_uv2,
                          omi_view_ang,
                          latitude,
                          longitude,
                          pressure,
                          utc,
                          time,
                          land_flag,
                          day_night_flag,
                          target_id,
                          doy,
                          col,
                          col_t,
                          col_ut,
                          col_error,
                          col_dry_air,
                          x,
                          x_prior,
                          species))


class _ReadL2Standard ():
    ##############
    # This creates a class for the TROPESS L2 Standard data.
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # self : an empty object/class; an object with the Standard TROPESS Products
    ##############
    def __init__(self,
                 number_of_files,
                 global_attrs,
                 date,
                 airs_granule,
                 airs_along_index,
                 airs_cross_index,
                 airs_view_ang,
                 omi_along_index,
                 omi_cross_index_uv1,
                 omi_cross_index_uv2,
                 omi_view_ang,
                 latitude,
                 longitude,
                 pressure,
                 utc,
                 time,
                 land_flag,
                 day_night_flag,
                 target_id,
                 doy,
                 x,
                 x_prior,
                 ak,
                 signal_dof,
                 x_error,
                 x_test,
                 air_density,
                 species):

        self.number_of_files = number_of_files
        self.global_attrs = global_attrs

        self.date = date
        self.airs_granule = airs_granule
        self.airs_along_index = airs_along_index
        self.airs_cross_index = airs_cross_index
        self.airs_view_ang = airs_view_ang
        self.omi_along_index = omi_along_index
        self.omi_cross_index_uv1 = omi_cross_index_uv1
        self.omi_cross_index_uv2 = omi_cross_index_uv2
        self.omi_view_ang = omi_view_ang
        self.latitude = latitude
        self.longitude = longitude
        self.pressure = pressure
        self.utc = utc
        self.time = time
        self.land_flag = land_flag
        self.day_night_flag = day_night_flag
        self.target_id = target_id
        self.doy = doy

        self.x = x
        self.x_prior = x_prior
        self.ak = ak
        self.signal_dof = signal_dof
        self.x_error = x_error
        self.x_test = x_test
        self.air_density = air_density

        self.species = species


def read_l2standard(files=None,
                    verbose=0):
    ##############
    # Read (multiple) TROPESS L2 Standard file(s).
    #
    # Parameters
    # ---------
    # file: string; full path to a CrIS L2 Standard file
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # _ReadL2Standard : an object/class; an object with the CrIS TROPESS Standard data
    ##############

    # Define species.
    species = "O3"

    # Define arrays
    number_of_files = len(files)
    global_attrs = {}

    n_rows = 15000*len(files)
    n_col = 26

    date = np.zeros((n_rows), dtype=int)
    airs_granule = np.zeros((n_rows), dtype=np.int32)
    airs_along_index = np.zeros((n_rows), dtype=np.int32)
    airs_cross_index = np.zeros((n_rows), dtype=np.int32)
    airs_view_ang = np.zeros((n_rows), dtype=np.float32)
    omi_along_index = np.zeros((n_rows), dtype=np.int32)
    omi_cross_index_uv1 = np.zeros((n_rows), dtype=np.int32)
    omi_cross_index_uv2 = np.zeros((n_rows), dtype=np.int32)
    omi_view_ang = np.zeros((n_rows), dtype=np.float32)
    latitude = np.zeros((n_rows), dtype=np.float32)
    longitude = np.zeros((n_rows), dtype=np.float32)
    pressure = np.zeros((n_rows, n_col), dtype=np.float32)
    utc = np.zeros((n_rows), dtype=np.float32)
    time = np.zeros((n_rows), dtype=np.float32)
    land_flag = np.zeros((n_rows), dtype=np.int32)
    day_night_flag = np.zeros((n_rows), dtype=np.int32)
    target_id = np.zeros((n_rows), dtype=np.int32)
    doy = np.zeros((n_rows), dtype=np.int32)

    x = np.zeros((n_rows, n_col), dtype=np.float32)  # vmr in ppbv
    x_prior = np.zeros((n_rows, n_col), dtype=np.float32)  # vmr in ppbv
    ak = np.zeros((n_rows, n_col, n_col), dtype=np.float32)
    signal_dof = np.zeros((n_rows), dtype=np.float32)
    x_error = np.zeros((n_rows, n_col), dtype=np.float32)
    x_test = np.zeros((n_rows, n_col), dtype=np.float32)  # vmr in ppbv
    air_density = np.zeros((n_rows, n_col), dtype=np.float32)  # mol m-3

    count = 0
    for i_files in range(0, number_of_files):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files])

        # Fill global attributes
        global_attrs[i_files] = data.global_attrs

        # Find length of the variables in this file
        l = len(data.values['/x'][:])

        # Fill variables
        pos_date = files[i_files].find(
            'TROPESS_AIRS-Aqua_OMI-Aura_L2_Standard_')
        pos_date2 = files[i_files].find('_', pos_date+40, -1)
        date[count:count+l] = int(files[i_files][pos_date2+1:pos_date2+5] +
                                  files[i_files][pos_date2+5:pos_date2+7] +
                                  files[i_files][pos_date2+7:pos_date2+9])
        if '/geolocation/airs_granule' in data.datasets:
            airs_granule[count:count +
                         l] = data.values['/geolocation/airs_granule'][:]
        airs_along_index[count:count +
                         l] = data.values['/geolocation/airs_atrack'][:]
        airs_cross_index[count:count +
                         l] = data.values['/geolocation/airs_xtrack'][:]
        airs_view_ang[count:count +
                      l] = data.values['/geolocation/airs_view_ang'][:]
        omi_along_index[count:count +
                        l] = data.values['/geolocation/omi_atrack'][:]
        omi_cross_index_uv1[count:count +
                            l] = data.values['/geolocation/omi_xtrack_uv1'][:]
        omi_cross_index_uv2[count:count +
                            l] = data.values['/geolocation/omi_xtrack_uv2'][:]
        omi_view_ang[count:count +
                     l] = data.values['/geolocation/omi_view_ang'][:]
        latitude[count:count+l] = data.values['/latitude'][:]
        longitude[count:count+l] = data.values['/longitude'][:]
        pressure[count:count+l] = data.values['/pressure'][:]
        dummy_utc = data.values['/datetime_utc'][:, 3] + \
            data.values['/datetime_utc'][:, 4]/60 + \
            data.values['/datetime_utc'][:, 5]/3600
        utc[count:count+l] = dummy_utc
        time[count:count+l] = data.values['/time'][:]
        land_flag[count:count +
                  l] = data.values['/geophysical/land_flag'][:]
        day_night_flag[count:count +
                       l] = data.values['/geophysical/day_night_flag'][:]
        if '/target_id' in data.datasets:
            target_id[count:count+l] = data.values['/target_id'][:]
        doy[count:count+l] = date_to_julian_day(int(files[i_files][pos_date2+1:pos_date2+5]),
                                                int(files[i_files]
                                                    [pos_date2+5:pos_date2+7]),
                                                int(files[i_files][pos_date2+7:pos_date2+9]))
        x[count:count+l, :] = data.values['/x'][:, :]
        x_prior[count:count+l, :] = data.values['/observation_ops/xa'][:]
        ak[count:count+l, :] = data.values['/observation_ops/averaging_kernel'][:, :]
        signal_dof[count:count +
                   l] = data.values['/observation_ops/signal_dof'][:]
        x_error[count:count+l, :] = np.diagonal(data.values['/observation_ops/observation_error'][:, :, :],
                                                axis1=1,
                                                axis2=2)
        x_test[count:count+l, :] = data.values['/observation_ops/x_test'][:]
        if '/retrieval/air_density' in data.datasets:
            air_density[count:count+l,
                        :] = data.values['/retrieval/air_density'][:, :]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    airs_granule = airs_granule[0:count]
    airs_along_index = airs_along_index[0:count]
    airs_cross_index = airs_cross_index[0:count]
    airs_view_ang = airs_view_ang[0:count]
    omi_along_index = omi_along_index[0:count]
    omi_cross_index_uv1 = omi_cross_index_uv1[0:count]
    omi_cross_index_uv2 = omi_cross_index_uv2[0:count]
    omi_view_ang = omi_view_ang[0:count]
    latitude = latitude[0:count]
    longitude = longitude[0:count]
    pressure = pressure[0:count]
    utc = utc[0:count]
    time = time[0:count]
    land_flag = land_flag[0:count]
    day_night_flag = day_night_flag[0:count]
    target_id = target_id[0:count]
    doy = doy[0:count]

    x = x[0:count]
    x_prior = x_prior[0:count]
    ak = ak[0:count]
    signal_dof = signal_dof[0:count]
    x_error = x_error[0:count]
    x_test = x_test[0:count]
    air_density = air_density[0:count]
    
    if np.std(airs_granule) == 0:
        airs_granule = None
    if np.std(target_id) == 0:
        target_id = None
    if np.std(air_density) == 0:
        air_density = None

    # Return data
    return(_ReadL2Standard(number_of_files,
                           global_attrs,
                           date,
                           airs_granule,
                           airs_along_index,
                           airs_cross_index,
                           airs_view_ang,
                           omi_along_index,
                           omi_cross_index_uv1,
                           omi_cross_index_uv2,
                           omi_view_ang,
                           latitude,
                           longitude,
                           pressure,
                           utc,
                           time,
                           land_flag,
                           day_night_flag,
                           target_id,
                           doy,
                           x,
                           x_prior,
                           ak,
                           signal_dof,
                           x_error,
                           x_test,
                           air_density,
                           species))
