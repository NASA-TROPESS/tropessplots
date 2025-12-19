"""
Title        :  cris.py
What is it   :  Routines to read and write CrIS data
Includes     :  read_cris_l1b()
                read_l2muses()
                read_l2lite()
                read_l2rad()
                read_l2summary()
                read_l2standard()
Author        : Frank Werner
Date          : 20240722
Modf          : 20240723: added automatic species detection for read_l2standard()
                20240913: added land_flag, cod, cqa, res_mean, res_rms to read_l2muses()
                20241205: added species to each class object; added number_of_files,
                          and global_attrs to read_l2summary() and read_l2standard()
                20250107: changed T to TATM in read_l2standard()
                20250123: added pixel_corners keyword to read_cris_l1b()
                20250125: added initial to read_l2muses(), added 
                          "# Define datasets" section to read_l2summary()
                          similar to read_l2standard()
                20250213: added species specific data_sets to read_l2summary()
                          and read_l2standard()
                20250606: added 'land_flag' variable to read_l2summary() and read_l2standard()
                20250614: added the ability to read SNPP CRIS data
                20250626: added '/observation_ops/signal_dof' to TATM in read_l2standard()
                20250905: added '/observation_ops/averaging_kernel', '/observation_ops/observation_error',
                          and '/observation_ops/xa' to TATM in read_l2standard(), added HDO 
                          transformations to read_l2lite() using hdo_specific,
                          added x, x_prior, and x_error to read_l2lite()
                20250914: added x_prior_hdohdo and x_prior_hdoh2o to read_l2lite()
                20250919: added full ak and col_air to read_l2lite()

                               
"""
# Import modules
# =======================================
import numpy as np

from tropessplots.shared.read_nc import read_nc
from tropessplots.shared.date_to_julian_day import date_to_julian_day
from tropessplots.calc.cris_pixel_corners import cris_pixel_corners
from tropessplots.calc.hdo_specific import x_species, xa_species, averaging_kernel, observation_error


# Functions and classes
# =======================================
class _ReadL1b ():
    ##############
    # This creates a class for the CrIS L1B data.
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # self : an object/class; an object with the CrIS L1B data
    ##############
    def __init__(self,
                 date,
                 granule,
                 fov,
                 rad_lw,
                 rad_mw,
                 rad_sw,
                 latitude,
                 longitude,
                 latitude_bnds,
                 longitude_bnds,
                 pixel_corners,
                 utc,
                 sol_zen_ang,
                 day_night_flag,
                 surf_alt,
                 view_ang,
                 doy):

        self.date = date
        self.granule = granule
        self.fov = fov
        self.rad_lw = rad_lw
        self.rad_mw = rad_mw
        self.rad_sw = rad_sw
        self.latitude = latitude
        self.longitude = longitude
        self.latitude_bnds = latitude_bnds
        self.longitude_bnds = longitude_bnds
        self.utc = utc
        self.sol_zen_ang = sol_zen_ang
        self.day_night_flag = day_night_flag
        self.surf_alt = surf_alt
        self.view_ang = view_ang
        self.doy = doy
        
        if pixel_corners is True:
            dummy = cris_pixel_corners(l1b=self)
            self.latitude_pixels = dummy[0]
            self.longitude_pixels = dummy[1]
        else:
            self.latitude_pixels = None
            self.longitude_pixels = None


def read_l1b(files=None,
             pixel_corners=False,
             verbose=0):
    ##############
    # This reads CrIS L1B files.
    #
    # Parameters
    # ---------
    # file: string; full path to a CrIS L1B file
    # pixel_corners: boolean; calculate pixel corners for each pixel and granule
    # pixel_corners_mode: string; if 'center_fov', pixel corners are based on the center FOV
    #                               of each of the 45x30 pixels.
    #                               If 'full_granule', pixel corners are 
    #                               calculated per granule and should align nicely.
    #                               I think 'center_fov' is closer to the truth;
    #                               'full_granule' might look nicer in plots.
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # _ReadL1b : an object/class; an object with the CrIS L1B data
    ##############

    # Define variables
    date = np.zeros((len(files), 45, 30, 9), dtype=int)
    granule = np.zeros((len(files), 45, 30, 9), dtype=int)
    fov = np.zeros((len(files), 45, 30, 9), dtype=int)
    rad_lw = np.zeros((len(files), 45, 30, 9, 717), dtype=np.float32)
    rad_mw = np.zeros((len(files), 45, 30, 9, 869), dtype=np.float32)
    rad_sw = np.zeros((len(files), 45, 30, 9, 637), dtype=np.float32)
    latitude = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    longitude = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    latitude_bnds = np.zeros((len(files), 45, 30, 9, 8), dtype=np.float32)
    longitude_bnds = np.zeros((len(files), 45, 30, 9, 8), dtype=np.float32)
    utc = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    sol_zen_ang = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    day_night_flag = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    surf_alt = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    view_ang = np.zeros((len(files), 45, 30, 9), dtype=np.float32)
    doy = np.zeros((len(files), 45, 30, 9), dtype=int)

    # Loop over all files
    for i_files in range(0, len(files)):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=['/rad_lw',
                                                            '/rad_mw',
                                                            '/rad_sw',
                                                            '/lat',
                                                            '/lon',
                                                            '/lat_bnds',
                                                            '/lon_bnds',
                                                            '/obs_time_utc',
                                                            '/sol_zen',
                                                            '/surf_alt',
                                                            '/view_ang'])

        # Fill variables
        pos_granule = files[i_files].find('.g')
        pos_granule2 = files[i_files].find('.', pos_granule+2, -1)
        date[i_files, :, :, :] = int(files[i_files][pos_granule-17:pos_granule-9])
        granule[i_files, :, :, :] = int(files[i_files][pos_granule+2:pos_granule2])
        fov[i_files, :, :, :] = np.reshape(np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8], len(
            data.values['/lat'][:, 0, 0])*len(data.values['/lat'][0, :, 0])), (45, 30, 9))
        rad_lw[i_files, :, :, :, :] = data.values['/rad_lw'][:, :, :, :]
        rad_mw[i_files, :, :, :, :] = data.values['/rad_mw'][:, :, :, :]
        rad_sw[i_files, :, :, :, :] = data.values['/rad_sw'][:, :, :, :]
        latitude[i_files, :, :, :] = data.values['/lat'][:, :, :]
        longitude[i_files, :, :, :] = data.values['/lon'][:, :, :]
        latitude_bnds[i_files, :, :, :, :] = data.values['/lat_bnds'][:, :, :]
        longitude_bnds[i_files, :, :, :, :] = data.values['/lon_bnds'][:, :, :]
        dummy_utc = data.values['/obs_time_utc'][:, :, 3]+data.values['/obs_time_utc'][:,
                                                                                       :, 4]/60 + data.values['/obs_time_utc'][:, :, 5]/3600
        utc[i_files, :, :, :] = np.reshape(np.tile(dummy_utc, 9), (45, 30, 9))
        sol_zen_ang[i_files, :, :, :] = data.values['/sol_zen'][:, :, :]
        day_night_flag_dummy = np.zeros_like(data.values['/sol_zen'][:, :, :])
        day_night_flag_dummy[np.where(
            (data.values['/sol_zen'][:, :, :] >= 0) & (data.values['/sol_zen'][:, :, :] <= 90))] = 1
        day_night_flag[i_files, :, :, :] = day_night_flag_dummy
        surf_alt[i_files, :, :, :] = data.values['/surf_alt'][:, :, :]
        view_ang[i_files, :, :, :] = data.values['/view_ang'][:, :, :]
        string_find = 'SNDR.J1.CRIS.'
        pos_date = files[i_files].find(string_find)
        if pos_date == -1:
            string_find = 'SNDR.SNPP.CRIS.'
            pos_date = files[i_files].find(string_find)
        doy[i_files, :, :, :] = date_to_julian_day(int(files[i_files][pos_date+len(string_find):pos_date+len(string_find)+4]),
                                                   int(files[i_files]
                                                       [pos_date+len(string_find)+4:pos_date+len(string_find)+6]),
                                                   int(files[i_files][pos_date+len(string_find)+6:pos_date+len(string_find)+8]))

    # Apply radiance scale factor
    rad_lw = rad_lw*1e-7
    rad_mw = rad_mw*1e-7
    rad_sw = rad_sw*1e-7

    # Return data
    return(_ReadL1b(date,
                    granule,
                    fov,
                    rad_lw,
                    rad_mw,
                    rad_sw,
                    latitude,
                    longitude,
                    latitude_bnds,
                    longitude_bnds,
                    pixel_corners,
                    utc,
                    sol_zen_ang,
                    day_night_flag,
                    surf_alt,
                    view_ang,
                    doy))


class _ReadL2Muses ():
    ##############
    # This creates a class for the MUSES L2_Products.
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # self : an empty object/class; an object with the MUSES L2_Products
    ##############
    def __init__(self,
                 date,
                 granule,
                 fov,
                 along_index,
                 cross_index,
                 latitude,
                 longitude,
                 utc,
                 day_night_flag,
                 land_flag,
                 view_ang,
                 doy,
                 quality,
                 dof,
                 ret_col,
                 ret_colerror,
                 ret_colprior,
                 ret_prior,
                 initial,
                 cod,
                 cqa,
                 res_mean,
                 res_rms,
                 species):

        self.date = date
        self.granule = granule
        self.fov = fov
        self.along_index = along_index
        self.cross_index = cross_index
        self.latitude = latitude
        self.longitude = longitude
        self.utc = utc
        self.day_night_flag = day_night_flag
        self.land_flag = land_flag
        self.view_ang = view_ang
        self.doy = doy

        self.quality = quality
        self.dof = dof
        self.ret_col = ret_col
        self.ret_colerror = ret_colerror
        self.ret_colprior = ret_colprior
        self.ret_prior = ret_prior
        self.initial = initial
        
        self.cod = cod
        self.cqa = cqa
        self.res_mean = res_mean
        self.res_rms = res_rms
        
        self.species = species


def read_l2muses(files=None,
                 verbose=0):
    ##############
    # Read (multiple) MUSES L2 Products file(s).
    #
    # Parameters
    # ---------
    # file: string; full path to a CrIS L2 MUSES file
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # _ReadL2Muses : an object/class; an object with the CrIS L2 MUSES data
    ##############

    # Define species.
    # Note that this code will break if you
    # mix Standard files from different species
    string_find = 'CRIS-JPSS-1_L2-'
    pos_find = files[0].find(string_find)
    if pos_find == -1:
        string_find = 'CRIS_L2-'
        pos_find = files[0].find(string_find)
    pos2_find = files[0].find('-',
                              pos_find+len(string_find),
                              -1)
    species = files[0][pos_find+len(string_find):pos2_find]

    # Define arrays
    n_rows = 49000*len(files)

    date = np.zeros((n_rows), dtype=int)
    granule = np.zeros((n_rows), dtype=np.int32)
    fov = np.zeros((n_rows), dtype=np.int32)
    along_index = np.zeros((n_rows), dtype=np.int32)
    cross_index = np.zeros((n_rows), dtype=np.int32)
    latitude = np.zeros((n_rows), dtype=np.float32)
    longitude = np.zeros((n_rows), dtype=np.float32)
    utc = np.zeros((n_rows), dtype=np.float32)
    day_night_flag = np.zeros((n_rows), dtype=np.int32)
    land_flag = np.zeros((n_rows), dtype=np.int32)
    view_ang = np.zeros((n_rows), dtype=np.float32)
    doy = np.zeros((n_rows), dtype=np.int32)

    quality = np.zeros((n_rows), dtype=np.int32)
    dof = np.zeros((n_rows, 5), dtype=np.float32)
    ret_col = np.zeros((n_rows, 5), dtype=np.float64)
    ret_colerror = np.zeros((n_rows, 5), dtype=np.float64)
    ret_colprior = np.zeros((n_rows, 5), dtype=np.float64)
    ret_prior = np.zeros((n_rows, 67), dtype=np.float32)
    initial = np.zeros((n_rows, 67), dtype=np.float32)
    
    cod = np.zeros((n_rows), dtype=np.float32)
    cqa = np.zeros((n_rows), dtype=np.int32)
    res_mean = np.zeros((n_rows), dtype=np.float32)
    res_rms = np.zeros((n_rows), dtype=np.float32)

    count = 0
    for i_files in range(0, len(files)):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=['/Geolocation/CrIS_Granule',
                                                            '/Geolocation/CrIS_Pixel_Index',
                                                            '/Geolocation/CrIS_Atrack_Index',
                                                            '/Geolocation/CrIS_Xtrack_Index',
                                                            '/Latitude',
                                                            '/Longitude',
                                                            '/UT_Hour',
                                                            '/DayNightFlag',
                                                            '/LandFlag',
                                                            '/Geolocation/PointingAngle_CrIS',
                                                            '/Quality',
                                                            '/Characterization/Column_DOFS',
                                                            '/Retrieval/Column',
                                                            '/Characterization/Column_Error',
                                                            '/Retrieval/Column_Prior',
                                                            '/ConstraintVector',
                                                            '/Characterization/Initial',
                                                            
                                                            '/Retrieval/AverageCloudEffOpticalDepth',
                                                            '/Characterization/CloudVariability_QA',
                                                            '/Characterization/RadianceResidualMean',
                                                            '/Characterization/RadianceResidualRMS'])

        # Find length of the variables in this file
        l = len(data.values['/Geolocation/CrIS_Granule'][:])

        # Fill variables
        pos_date = files[i_files].find(string_find)
        pos_date2 = files[i_files].find('_', pos_date+len(string_find), -1)
        date[count:count+l] = int(files[i_files][pos_date2+1:pos_date2+5] +
                                  files[i_files][pos_date2+6:pos_date2+8] +
                                  files[i_files][pos_date2+9:pos_date2+11])
        granule[count:count+l] = data.values['/Geolocation/CrIS_Granule'][:]
        fov[count:count+l] = data.values['/Geolocation/CrIS_Pixel_Index'][:]
        along_index[count:count +
                    l] = data.values['/Geolocation/CrIS_Atrack_Index'][:]
        cross_index[count:count +
                    l] = data.values['/Geolocation/CrIS_Xtrack_Index'][:]
        latitude[count:count+l] = data.values['/Latitude'][:]
        longitude[count:count+l] = data.values['/Longitude'][:]
        utc[count:count+l] = data.values['/UT_Hour'][:]
        day_night_flag[count:count+l] = data.values['/DayNightFlag'][:]
        land_flag[count:count+l] = data.values['/LandFlag'][:]
        view_ang[count:count+l] = data.values['/Geolocation/PointingAngle_CrIS'][:]
        doy[count:count+l] = date_to_julian_day(int(files[i_files][pos_date2+1:pos_date2+5]),
                                                int(files[i_files]
                                                    [pos_date2+6:pos_date2+8]),
                                                int(files[i_files][pos_date2+9:pos_date2+11]))

        quality[count:count+l] = data.values['/Quality'][:]
        dof[count:count+l, :] = data.values['/Characterization/Column_DOFS'][:, :]
        ret_col[count:count+l, :] = data.values['/Retrieval/Column'][:, :]
        ret_colerror[count:count+l,
                     :] = data.values['/Characterization/Column_Error'][:, :]
        ret_colprior[count:count+l,
                     :] = data.values['/Retrieval/Column_Prior'][:, :]
        ret_prior[count:count+l, :] = data.values['/ConstraintVector'][:, :]
        initial[count:count+l, :] = data.values['/Characterization/Initial'][:, :]
        
        cod[count:count+l] = data.values['/Retrieval/AverageCloudEffOpticalDepth'][:]
        cqa[count:count+l] = data.values['/Characterization/CloudVariability_QA'][:]
        res_mean[count:count+l] = data.values['/Characterization/RadianceResidualMean'][:]
        res_rms[count:count+l] = data.values['/Characterization/RadianceResidualRMS'][:]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    granule = granule[0:count]
    fov = fov[0:count]
    along_index = along_index[0:count]
    cross_index = cross_index[0:count]
    latitude = latitude[0:count]
    longitude = longitude[0:count]
    utc = utc[0:count]
    day_night_flag = day_night_flag[0:count]
    land_flag = land_flag[0:count]
    view_ang = view_ang[0:count]
    doy = doy[0:count]

    quality = quality[0:count]
    dof = dof[0:count]
    ret_col = ret_col[0:count]
    ret_colerror = ret_colerror[0:count]
    ret_colprior = ret_colprior[0:count]
    ret_prior = ret_prior[0:count]
    initial = initial[0:count]
    
    cod = cod[0:count]
    cqa = cqa[0:count]
    res_mean = res_mean[0:count]
    res_rms = res_rms[0:count]

    # Return data
    return(_ReadL2Muses(date,
                        granule,
                        fov,
                        along_index,
                        cross_index,
                        latitude,
                        longitude,
                        utc,
                        day_night_flag,
                        land_flag,
                        view_ang,
                        doy,
                        quality,
                        dof,
                        ret_col,
                        ret_colerror,
                        ret_colprior,
                        ret_prior,
                        initial,
                        cod,
                        cqa,
                        res_mean,
                        res_rms,
                        species))


class _ReadL2Lite ():
    ##############
    # This creates a class for the MUSES L2_Products_Lite.
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # self : an empty object/class; an object with the MUSES L2_Products_Lite
    ##############
    def __init__(self,
                 date,
                 pressure,
                 col_air,
                 ak,
                 akdiag,
                 x,
                 x_prior,
                 x_prior_hdohdo,
                 x_prior_hdoh2o,
                 x_error,
                 surf_alt,
                 surf_temp,
                 trop_press,
                 species):

        self.date = date
        self.pressure = pressure
        self.col_air = col_air
        self.ak = ak
        self.akdiag = akdiag
        self.x = x
        self.x_prior = x_prior
        self.x_prior_hdohdo = x_prior_hdohdo
        self.x_prior_hdoh2o = x_prior_hdoh2o
        self.x_error = x_error
        self.surf_alt = surf_alt
        self.surf_temp = surf_temp
        self.trop_press = trop_press
        
        self.species = species


def read_l2lite(files=None,
                verbose=0):
    ##############
    # Read (multiple) MUSES L2_Products_Lite file(s).
    #
    # Parameters
    # ---------
    # file: string; full path to a CrIS L2_Products_Lite file
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # _ReadL2Lite : an object/class; an object with the CrIS L2_Products_Lite data
    ##############

    # Define species.
    # Note that this code will break if you
    # mix Standard files from different species
    string_find = 'CRIS-JPSS-1_L2-'
    pos_find = files[0].find(string_find)
    if pos_find == -1:
        string_find = 'CRIS_L2-'
        pos_find = files[0].find(string_find)
    pos2_find = files[0].find('-',
                              pos_find+len(string_find),
                              -1)
    species = files[0][pos_find+len(string_find):pos2_find]

    # Define arrays
    n_rows = 49000*len(files)
    
    # Define data sets to be read
    data_sets=['/Pressure',
               '/Retrieval/Column_Air',
               '/AveragingKernel',
               '/Characterization/AveragingKernelDiagonal',
               '/ConstraintVector',
               '/Species',
               '/ObservationErrorCovariance',
               '/SurfaceAltitude',
               '/Retrieval/SurfaceTemperature',
               '/Characterization/TropopausePressure']

    # Define arrays
    date = np.zeros((n_rows), dtype=int)
    if species == 'CO':
        n_lvls = 14
    if species == 'NH3':
        n_lvls = 15
    if species == 'O3' or species == 'CH4' or species == 'N2O':
        n_lvls = 26
    if species == 'H2O' or species == 'HDO':
        n_lvls = 17
    if species == 'PAN':
        n_lvls = 16
    if species == 'TATM':
        n_lvls = 31
    pressure = np.zeros((n_rows, n_lvls), dtype=np.float32)
    col_air = np.zeros((n_rows, 5), dtype=np.float32)
    ak = np.zeros((n_rows, n_lvls, n_lvls), dtype=np.float32)
    akdiag = np.zeros((n_rows, n_lvls), dtype=np.float32)
    x = np.zeros((n_rows, n_lvls), dtype=np.float32)
    x_prior = np.zeros((n_rows, n_lvls), dtype=np.float32)
    if species != 'HDO':
        x_prior_hdohdo = None
        x_prior_hdoh2o = None
    else:
        x_prior_hdohdo = np.zeros((n_rows, n_lvls), dtype=np.float32)
        x_prior_hdoh2o = np.zeros((n_rows, n_lvls), dtype=np.float32)
    x_error = np.zeros((n_rows, n_lvls), dtype=np.float32)
    surf_alt = np.zeros((n_rows), dtype=np.float32)
    surf_temp = np.zeros((n_rows), dtype=np.float32)
    trop_press = np.zeros((n_rows), dtype=np.float32)

    count = 0
    for i_files in range(0, len(files)):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=data_sets)

        # Find length of the variables in this file
        l = len(
            data.values['/Characterization/AveragingKernelDiagonal'][:])

        # Fill variables
        pos_date = files[i_files].find(string_find)
        pos_date2 = files[i_files].find('_', pos_date+len(string_find), -1)
        date[count:count+l] = int(files[i_files][pos_date2+1:pos_date2+5] +
                                  files[i_files][pos_date2+6:pos_date2+8] +
                                  files[i_files][pos_date2+9:pos_date2+11])
        pressure[count:count+l,
               :] = data.values['/Pressure'][:, :n_lvls]
        col_air[count:count+l,
               :] = data.values['/Retrieval/Column_Air'][:, :]
        if species != 'HDO':
            ak[count:count+l,
                   :, :] = data.values['/AveragingKernel'][:, :, :]
            akdiag[count:count+l,
                   :] = data.values['/Characterization/AveragingKernelDiagonal'][:, :]
            x[count:count+l,
                   :] = data.values['/Species'][:, :]
            x_prior[count:count+l,
                   :] = data.values['/ConstraintVector'][:, :]
            x_error[count:count+l,
                   :] = np.diagonal(data.values['/ObservationErrorCovariance'][:, :, :],
                                    axis1=1,
                                    axis2=2)
        else:
            ak[count:count+l,
                   :, :] = averaging_kernel(data.values['/AveragingKernel'][:, :, :])
            akdiag[count:count+l,
                   :] = np.diagonal(averaging_kernel(data.values['/AveragingKernel'][:, :, :]),
                                    axis1=1,
                                    axis2=2)
            x[count:count+l,
                   :] = x_species(data.values['/Species'][:, :])
            dummy1, dummy2, dummy3 = xa_species(data.values['/ConstraintVector'][:, :])
            x_prior[count:count+l,
                   :] = dummy1
            x_prior_hdohdo[count:count+l,
                   :] = dummy2
            x_prior_hdoh2o[count:count+l,
                   :] = dummy3
            x_error[count:count+l,
                   :] = np.diagonal(observation_error(data.values['/ObservationErrorCovariance'][:, :, :]),
                                    axis1=1,
                                    axis2=2)
        surf_alt[count:count+l] = data.values['/SurfaceAltitude'][:]
        surf_temp[count:count +
                  l] = data.values['/Retrieval/SurfaceTemperature'][:]
        trop_press[count:count +
                   l] = data.values['/Characterization/TropopausePressure'][:]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    pressure = pressure[0:count]
    col_air = col_air[0:count]
    ak = ak[0:count]
    akdiag = akdiag[0:count]
    x = x[0:count]
    x_prior = x_prior[0:count]
    if species == 'HDO':
        x_prior_hdohdo = x_prior_hdohdo[0:count]
        x_prior_hdoh2o = x_prior_hdoh2o[0:count]
    x_error = x_error[0:count]
    surf_alt = surf_alt[0:count]
    surf_temp = surf_temp[0:count]
    trop_press = trop_press[0:count]
    
    # Filter
    missing_value=9.969209968386869e+36
    ind = np.where(pressure==missing_value)
    pressure[ind] = -999.
    ind = np.where(col_air==missing_value)
    col_air[ind] = -999.
    ind = np.where(ak==missing_value)
    ak[ind] = -999.
    ind = np.where(akdiag==missing_value)
    akdiag[ind] = -999.
    ind = np.where(x==missing_value)
    x[ind] = -999.
    ind = np.where(x_prior==missing_value)
    x_prior[ind] = -999.
    if species == 'HDO':
        ind = np.where(x_prior_hdohdo==missing_value)
        x_prior_hdohdo[ind] = -999.
        ind = np.where(x_prior_hdoh2o==missing_value)
        x_prior_hdoh2o[ind] = -999.
    ind = np.where(x_error==missing_value)
    x_error[ind] = -999.

    # Return data
    return(_ReadL2Lite(date,
                       pressure,
                       col_air,
                       ak,
                       akdiag,
                       x,
                       x_prior,
                       x_prior_hdohdo,
                       x_prior_hdoh2o,
                       x_error,
                       surf_alt,
                       surf_temp,
                       trop_press,
                       species))


class _ReadL2Rad ():
    ##############
    # This creates a class for the MUSES L2_Radiance.
    #
    # Parameters
    # ---------
    # None
    #
    # Returns
    # -------
    # self : an empty object/class; an object with the MUSES L2_Radiance
    ##############
    def __init__(self,
                 date,
                 rad,
                 freq,
                 rad_full,
                 freq_full,
                 latitude,
                 longitude):

        self.date = date
        self.rad = rad
        self.freq = freq
        self.rad_full = rad_full
        self.freq_full = freq_full
        self.latitude = latitude
        self.longitude = longitude


def read_l2rad(files=None,
               verbose=0):
    ##############
    # Read (multiple) MUSES L2 Radiance file(s).
    #
    # Parameters
    # ---------
    # file: string; full path to a CrIS L2 MUSES file
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # _ReadL2Rad : an object/class; an object with the CrIS L2 Radiance data
    ##############

    # Define species.
    # Note that this code will break if you
    # mix Radiance files from different species
    string_find = 'Products_Radiance-'
    pos_find = files[0].find(string_find)
    pos2_find = files[0].find('_',
                              pos_find+len(string_find),
                              -1)
    species = files[0][pos_find+len(string_find):pos2_find]

    # Define arrays
    n_rows = 49000*len(files)

    date = np.zeros((n_rows), dtype=int)
    if species == 'CO':
        rad = np.zeros((n_rows, 31), dtype=np.float32)
        freq = np.zeros((n_rows, 31), dtype=np.float32)
    if species == 'NH3':
        rad = np.zeros((n_rows, 11), dtype=np.float32)
        freq = np.zeros((n_rows, 11), dtype=np.float32)
    if species == 'O3' or species == 'H2O' or species == 'H2O,O3':
        rad = np.zeros((n_rows, 216), dtype=np.float32)
        freq = np.zeros((n_rows, 216), dtype=np.float32)
    if species == 'CH4' or species == 'TATM,H2O,HDO,N2O,CH4,O3,TSUR,CLOUDEXT-bar':
        rad = np.zeros((n_rows, 475), dtype=np.float32)
        freq = np.zeros((n_rows, 475), dtype=np.float32)
    if species == 'PAN':
        rad = np.zeros((n_rows, 14), dtype=np.float32)
        freq = np.zeros((n_rows, 14), dtype=np.float32)
    if species == 'N2O' or species == 'TATM' or species == 'TATM,H2O,HDO,N2O,CH4,O3,TSUR,CLOUDEXT-bar':
        rad = np.zeros((n_rows, 475), dtype=np.float32)
        freq = np.zeros((n_rows, 475), dtype=np.float32)
    rad_full = np.zeros((n_rows, 2223), dtype=np.float32)
    freq_full = np.zeros((n_rows, 2223), dtype=np.float32)
    latitude = np.zeros((n_rows), dtype=np.float32)
    longitude = np.zeros((n_rows), dtype=np.float32)

    count = 0
    for i_files in range(0, len(files)):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=['/RADIANCEOBSERVED',
                                                            '/FREQUENCY',
                                                            '/RADIANCEFULLBAND',
                                                            '/FREQUENCYFULLBAND',
                                                            '/LATITUDE',
                                                            '/LONGITUDE'])

        # Find length of the variables in this file
        l = len(data.values['/LATITUDE'][:])

        # Fill variables
        pos_date = files[i_files].find('.nc')
        date[count:count+l] = int(files[i_files][pos_date-10:pos_date-6] +
                                  files[i_files][pos_date-5:pos_date-3] +
                                  files[i_files][pos_date-2:pos_date])
        rad[count:count+l, :] = data.values['/RADIANCEOBSERVED'][:, :]
        freq[count:count+l, :] = data.values['/FREQUENCY'][:, :]
        rad_full[count:count+l,
                 :] = data.values['/RADIANCEFULLBAND'][:, :]
        freq_full[count:count+l,
                  :] = data.values['/FREQUENCYFULLBAND'][:, :]
        latitude[count:count+l] = data.values['/LATITUDE'][:]
        longitude[count:count+l] = data.values['/LONGITUDE'][:]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    rad = rad[0:count]
    freq = freq[0:count]
    rad_full = rad_full[0:count]
    freq_full = freq_full[0:count]
    latitude = latitude[0:count]
    longitude = longitude[0:count]

    # Return data
    return(_ReadL2Rad(date,
                      rad,
                      freq,
                      rad_full,
                      freq_full,
                      latitude,
                      longitude))


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
                 granule,
                 fov,
                 along_index,
                 cross_index,
                 latitude,
                 longitude,
                 pressure,
                 utc,
                 time,
                 land_flag,
                 day_night_flag,
                 view_ang,
                 doy,
                 col,
                 col_t,
                 col_ut,
                 col_error,
                 x_col,
                 x_col_error,
                 x_col_ft,
                 x_col_ft_error,
                 x_col_p,
                 x_col_p_error,
                 x,
                 x_prior,
                 xa_col_ft,
                 ak_col,
                 ak_x_col,
                 ak_x_col_ft,
                 ak_x_col_p,
                 species):

        self.number_of_files = number_of_files
        self.global_attrs = global_attrs
        
        self.date = date
        self.granule = granule
        self.fov = fov
        self.along_index = along_index
        self.cross_index = cross_index
        self.latitude = latitude
        self.longitude = longitude
        self.pressure = pressure
        self.utc = utc
        self.time = time
        self.land_flag = land_flag
        self.day_night_flag = day_night_flag
        self.view_ang = view_ang
        self.doy = doy

        self.col = col
        self.col_t = col_t
        self.col_ut = col_ut
        self.col_error = col_error
        self.x_col = x_col
        self.x_col_error = x_col_error
        self.x_col_ft = x_col_ft
        self.x_col_ft_error = x_col_ft_error
        self.x_col_p = x_col_p
        self.x_col_p_error = x_col_p_error
        self.x = x
        self.x_prior = x_prior
        self.xa_col_ft = xa_col_ft

        self.ak_col = ak_col
        self.ak_x_col = ak_x_col
        self.ak_x_col_ft = ak_x_col_ft
        self.ak_x_col_p = ak_x_col_p
        
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
    species = files[0].split('/')[-1].split('_')[4]

    # Define datasets
    if species == 'CO':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/col',
                     '/col_error',
                     '/x_col',
                     '/x_col_error',
                     '/x',
                     '/xa',
                     '/ak_col',
                     '/ak_x_col']
        n_col = 14
    if species == 'NH3':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/col',
                     '/col_error',
                     '/x',
                     '/xa',
                     '/ak_col']
        n_col = 15
    if species == 'O3':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/col',
                     '/col_t',
                     '/col_ut',
                     '/col_error',
                     '/x',
                     '/xa']
        n_col = 26
    if species == 'CH4':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/x_col_p',
                     '/x_col_p_error',
                     '/x',
                     '/xa',
                     '/ak_x_col_p']
        n_col = 26
    if species == 'PAN':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/x_col_ft',
                     '/x_col_ft_error',
                     '/xa_col_ft',
                     '/ak_x_col_ft']
        n_col = 16

    # Define arrays
    number_of_files = len(files)
    global_attrs = {}
    
    n_rows = 49000*len(files)

    date = np.zeros((n_rows), dtype=int)
    granule = np.zeros((n_rows), dtype=np.int32)
    fov = np.zeros((n_rows), dtype=np.int32)
    along_index = np.zeros((n_rows), dtype=np.int32)
    cross_index = np.zeros((n_rows), dtype=np.int32)
    latitude = np.zeros((n_rows), dtype=np.float32)
    longitude = np.zeros((n_rows), dtype=np.float32)
    pressure = np.zeros((n_rows, n_col), dtype=np.float32)
    utc = np.zeros((n_rows), dtype=np.float32)
    time = np.zeros((n_rows), dtype=np.float32)
    land_flag = np.zeros((n_rows), dtype=np.int32)
    day_night_flag = np.zeros((n_rows), dtype=np.int32)
    view_ang = np.zeros((n_rows), dtype=np.float32)
    doy = np.zeros((n_rows), dtype=np.int32)

    col = np.zeros((n_rows), dtype=np.float32) # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19], 'for_dobson_units_multiply_by': array([2241.1475], dtype=float32)
    col_t = np.zeros((n_rows), dtype=np.float32) # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19], 'for_dobson_units_multiply_by': array([2241.1475], dtype=float32)
    col_ut = np.zeros((n_rows), dtype=np.float32) # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19], 'for_dobson_units_multiply_by': array([2241.1475], dtype=float32)
    col_error = np.zeros((n_rows), dtype=np.float32) # 'mol m-2', 'for_molecules_per_cm2_multiply_by': [6.022141e+19]
    x_col = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv
    x_col_error = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv
    x_col_ft = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv
    x_col_ft_error = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv
    x_col_p = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv
    x_col_p_error = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv
    x = np.zeros((n_rows, n_col), dtype=np.float32) # vmr in ppbv
    x_prior = np.zeros((n_rows, n_col), dtype=np.float32) # vmr in ppbv
    xa_col_ft = np.zeros((n_rows), dtype=np.float32) # vmr in ppbv

    ak_col = np.zeros((n_rows, n_col), dtype=np.float32)
    ak_x_col = np.zeros((n_rows, n_col), dtype=np.float32)
    ak_x_col_ft = np.zeros((n_rows, n_col), dtype=np.float32)
    ak_x_col_p = np.zeros((n_rows, n_col), dtype=np.float32)

    count = 0
    for i_files in range(0, number_of_files):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=data_sets)
        
        # Fill global attributes
        global_attrs[i_files] = data.global_attrs

        # Find length of the variables in this file
        l = len(data.values['/geolocation/cris_granule'][:])

        # Fill variables
        pos_date = files[i_files].find('TROPESS_CrIS-JPSS1_L2_Summary_')
        pos_date2 = files[i_files].find('_', pos_date+30, -1)
        date[count:count+l] = int(files[i_files][pos_date2+1:pos_date2+5] +
                                  files[i_files][pos_date2+5:pos_date2+7] +
                                  files[i_files][pos_date2+7:pos_date2+9])
        granule[count:count+l] = data.values['/geolocation/cris_granule'][:]
        fov[count:count+l] = data.values['/geolocation/cris_fov'][:]
        along_index[count:count +
                    l] = data.values['/geolocation/cris_atrack'][:]
        cross_index[count:count +
                    l] = data.values['/geolocation/cris_xtrack'][:]
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
        view_ang[count:count+l] = data.values['/geolocation/cris_view_ang'][:]
        doy[count:count+l] = date_to_julian_day(int(files[i_files][pos_date2+1:pos_date2+5]),
                                                int(files[i_files]
                                                    [pos_date2+5:pos_date2+7]),
                                                int(files[i_files][pos_date2+7:pos_date2+9]))

        if '/col' in data_sets:
            col[count:count+l] = data.values['/col'][:]
        if '/col_t' in data_sets:
            col_t[count:count+l] = data.values['/col_t'][:]
        if '/col_ut' in data_sets:
            col_ut[count:count+l] = data.values['/col_ut'][:]
        if '/col_error' in data_sets:
             col_error[count:count+l] = data.values['/col_error'][:]
        if '/x_col' in data_sets:
            x_col[count:count+l] = data.values['/x_col'][:]
            x_col_error[count:count+l] = data.values['/x_col_error'][:]
        if '/x_col_ft' in data_sets:
            x_col_ft[count:count+l] = data.values['/x_col_ft'][:]
            x_col_ft_error[count:count+l] = data.values['/x_col_ft_error'][:]
        if '/x_col_p' in data_sets:
            x_col_p[count:count+l] = data.values['/x_col_p'][:]
            x_col_p_error[count:count+l] = data.values['/x_col_p_error'][:]
        if '/x' in data_sets:
            x[count:count+l, :] = data.values['/x'][:, :]
        if '/xa' in data_sets:
            x_prior[count:count+l, :] = data.values['/xa'][:]
        if '/xa_col_ft' in data_sets:
            xa_col_ft[count:count+l] = data.values['/xa_col_ft'][:]

        if '/ak_col' in data_sets:
            ak_col[count:count+l, :] = data.values['/ak_col'][:, :]
        if '/ak_x_col' in data_sets:
            ak_x_col[count:count+l, :] = data.values['/ak_x_col'][:]
        if '/ak_x_col_ft' in data_sets:
            ak_x_col_ft[count:count+l, :] = data.values['/ak_x_col_ft'][:]
        if '/ak_x_col_p' in data_sets:
            ak_x_col_p[count:count+l, :] = data.values['/ak_x_col_p'][:]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    granule = granule[0:count]
    fov = fov[0:count]
    along_index = along_index[0:count]
    cross_index = cross_index[0:count]
    latitude = latitude[0:count]
    longitude = longitude[0:count]
    pressure = pressure[0:count]
    utc = utc[0:count]
    time = time[0:count]
    land_flag = land_flag[0:count]
    day_night_flag = day_night_flag[0:count]
    view_ang = view_ang[0:count]
    doy = doy[0:count]

    if '/col' in data_sets:
        col = col[0:count]
    else:
        col = col[0:count]
    if '/col_t' in data_sets:
        col_t = col_t[0:count]
    else:
        col_t = None
    if '/col_ut' in data_sets:
        col_ut = col_ut[0:count]
    else:
        col_ut = None
    if '/col_error' in data_sets:
        col_error = col_error[0:count]
    else:
        col_error = None
    if '/x_col' in data_sets:
        x_col = x_col[0:count]
        x_col_error = x_col_error[0:count]
    else:
        x_col = None
        x_col_error = None
    if '/x_col_ft' in data_sets:
        x_col_ft = x_col_ft[0:count]
        x_col_ft_error = x_col_ft_error[0:count]
    else:
        x_col_ft = None
        x_col_ft_error = None
    if '/x_col_p' in data_sets:
        x_col_p = x_col_p[0:count]
        x_col_p_error = x_col_p_error[0:count]
    else:
        x_col_p = None
        x_col_p_error = None
    if '/x' in data_sets:
        x = x[0:count, :]
    else:
        x = x[0:count, :]
    if '/xa' in data_sets:
        x_prior = x_prior[0:count, :]
    else:
        x_prior = x_prior[0:count, :]
    if '/xa_col_ft' in data_sets:
        xa_col_ft = xa_col_ft[0:count]
    else:
        xa_col_ft = None

    if '/ak_col' in data_sets:
        ak_col = ak_col[0:count, :]
    else:
        ak_col = None
    if '/ak_x_col' in data_sets:
        ak_x_col = ak_x_col[0:count, :]
    else:
        ak_x_col = None
    if '/ak_x_col_ft' in data_sets:
        ak_x_col_ft = ak_x_col_ft[0:count, :]
    else:
        ak_x_col_ft = None
    if '/ak_x_col_p' in data_sets:
        ak_x_col_p = ak_x_col_p[0:count, :]
    else:
        ak_x_col_p = None

    # Return data
    return(_ReadL2Summary(number_of_files,
                          global_attrs,
                          date,
                          granule,
                          fov,
                          along_index,
                          cross_index,
                          latitude,
                          longitude,
                          pressure,
                          utc,
                          time,
                          land_flag,
                          day_night_flag,
                          view_ang,
                          doy,
                          col,
                          col_t,
                          col_ut,
                          col_error,
                          x_col,
                          x_col_error,
                          x_col_ft,
                          x_col_ft_error,
                          x_col_p,
                          x_col_p_error,
                          x,
                          x_prior,
                          xa_col_ft,
                          ak_col,
                          ak_x_col,
                          ak_x_col_ft,
                          ak_x_col_p,
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
                 granule,
                 fov,
                 along_index,
                 cross_index,
                 latitude,
                 longitude,
                 pressure,
                 utc,
                 time,
                 land_flag,
                 day_night_flag,
                 view_ang,
                 doy,
                 x,
                 x_error,
                 x_prior,
                 ak,
                 signal_dof,
                 species):

        self.number_of_files = number_of_files
        self.global_attrs = global_attrs
        
        self.date = date
        self.granule = granule
        self.fov = fov
        self.along_index = along_index
        self.cross_index = cross_index
        self.latitude = latitude
        self.longitude = longitude
        self.pressure = pressure
        self.utc = utc
        self.time = time
        self.land_flag = land_flag
        self.day_night_flag = day_night_flag
        self.view_ang = view_ang
        self.doy = doy

        self.x = x
        self.x_error = x_error
        self.x_prior = x_prior

        self.ak = ak
        self.signal_dof = signal_dof
        
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
    species = files[0].split('/')[-1].split('_')[4]

    # Define datasets
    if species == 'CO':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 14
    if species == 'NH3':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 15
    if species == 'O3':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 26
    if species == 'CH4':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 26
    if species == 'PAN':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 16
    if species == 'TATM':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 31
    if species == 'H2O':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x']
        n_col = 17
    if species == 'HDO':
        data_sets = ['/geolocation/cris_granule',
                     '/geolocation/cris_fov',
                     '/geolocation/cris_atrack',
                     '/geolocation/cris_xtrack',
                     '/latitude',
                     '/longitude',
                     '/pressure',
                     '/datetime_utc',
                     '/time',
                     '/geophysical/land_flag',
                     '/geophysical/day_night_flag',
                     '/geolocation/cris_view_ang',
                     '/observation_ops/averaging_kernel',
                     '/observation_ops/observation_error',
                     '/observation_ops/signal_dof',
                     '/observation_ops/xa',
                     '/x',
                     '/x_h2o']
        n_col = 17

    # Define arrays
    number_of_files = len(files)
    global_attrs = {}
    
    n_rows = 49000*len(files)

    date = np.zeros((n_rows), dtype=int)
    granule = np.zeros((n_rows), dtype=np.int32)
    fov = np.zeros((n_rows), dtype=np.int32)
    along_index = np.zeros((n_rows), dtype=np.int32)
    cross_index = np.zeros((n_rows), dtype=np.int32)
    latitude = np.zeros((n_rows), dtype=np.float32)
    longitude = np.zeros((n_rows), dtype=np.float32)
    pressure = np.zeros((n_rows, n_col), dtype=np.float32)
    utc = np.zeros((n_rows), dtype=np.float32)
    time = np.zeros((n_rows), dtype=np.float32)
    land_flag = np.zeros((n_rows), dtype=np.int32)
    day_night_flag = np.zeros((n_rows), dtype=np.int32)
    view_ang = np.zeros((n_rows), dtype=np.float32)
    doy = np.zeros((n_rows), dtype=np.int32)

    x = np.zeros((n_rows, n_col), dtype=np.float32) # vmr in ppbv
    x_error = np.zeros((n_rows, n_col), dtype=np.float32) # vmr in ppbv
    x_prior = np.zeros((n_rows, n_col), dtype=np.float32) # vmr in ppbv

    ak = np.zeros((n_rows, n_col, n_col), dtype=np.float32)
    signal_dof = np.zeros((n_rows), dtype=np.float32)

    count = 0
    for i_files in range(0, number_of_files):
        if verbose == 1:
            print('Reading file ', i_files+1, '/', len(files))
        # Read data sets from .nc file
        data = read_nc(file_name=files[i_files], data_sets=data_sets)
        
        # Fill global attributes
        global_attrs[i_files] = data.global_attrs

        # Find length of the variables in this file
        l = len(data.values['/geolocation/cris_granule'][:])

        # Fill variables
        pos_date = files[i_files].find('TROPESS_CrIS-JPSS1_L2_Standard_')
        pos_date2 = files[i_files].find('_', pos_date+31, -1)
        date[count:count+l] = int(files[i_files][pos_date2+1:pos_date2+5] +
                                  files[i_files][pos_date2+5:pos_date2+7] +
                                  files[i_files][pos_date2+7:pos_date2+9])
        granule[count:count+l] = data.values['/geolocation/cris_granule'][:]
        fov[count:count+l] = data.values['/geolocation/cris_fov'][:]
        along_index[count:count +
                    l] = data.values['/geolocation/cris_atrack'][:]
        cross_index[count:count +
                    l] = data.values['/geolocation/cris_xtrack'][:]
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
        view_ang[count:count+l] = data.values['/geolocation/cris_view_ang'][:]
        doy[count:count+l] = date_to_julian_day(int(files[i_files][pos_date2+1:pos_date2+5]),
                                                int(files[i_files]
                                                    [pos_date2+5:pos_date2+7]),
                                                int(files[i_files][pos_date2+7:pos_date2+9]))

        if '/x' in data.datasets:
            x[count:count+l, :] = data.values['/x'][:, :]
        else:
            x = None
        if '/observation_ops/observation_error' in data.datasets:
            x_error[count:count+l, :] = np.diagonal(data.values['/observation_ops/observation_error'][:, :, :],
                                                    axis1=1,
                                                    axis2=2)
        else:
            x_error = None
        if '/observation_ops/xa' in data.datasets:
            x_prior[count:count+l, :] = data.values['/observation_ops/xa'][:]
        else:
            x_prior = None

        if '/observation_ops/averaging_kernel' in data.datasets:
            ak[count:count+l, :, :] = data.values['/observation_ops/averaging_kernel'][:, :, :]
        else:
            ak = None
        if '/observation_ops/signal_dof' in data.datasets:
            signal_dof[count:count+l] = data.values['/observation_ops/signal_dof'][:]
        else:
            signal_dof = None
        #if '/x_h2o' in data.datasets:
            #x[count:count+l, :] *= data.values['/x_h2o'][:, :]
            #x_error[count:count+l, :] *= data.values['/x_h2o'][:, :]
            #x_prior[count:count+l, :] *= data.values['/x_h2o'][:, :]

        # Update the count
        count += l

    # Reduce the data set
    date = date[0:count]
    granule = granule[0:count]
    fov = fov[0:count]
    along_index = along_index[0:count]
    cross_index = cross_index[0:count]
    latitude = latitude[0:count]
    longitude = longitude[0:count]
    pressure = pressure[0:count]
    utc = utc[0:count]
    time = time[0:count]
    land_flag = land_flag[0:count]
    day_night_flag = day_night_flag[0:count]
    view_ang = view_ang[0:count]
    doy = doy[0:count]

    if x is not None:
        x = x[0:count, :]
    if x_error is not None:
        x_error = x_error[0:count, :]
    if x_prior is not None:
        x_prior = x_prior[0:count, :]

    if ak is not None:
        ak = ak[0:count, :, :]
    if signal_dof is not None:
        signal_dof = signal_dof[0:count]

    # Return data
    return(_ReadL2Standard(number_of_files,
                           global_attrs,
                           date,
                           granule,
                           fov,
                           along_index,
                           cross_index,
                           latitude,
                           longitude,
                           pressure,
                           utc,
                           time,
                           land_flag,
                           day_night_flag,
                           view_ang,
                           doy,
                           x,
                           x_error,
                           x_prior,
                           ak,
                           signal_dof,
                           species))
