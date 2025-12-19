"""
Title        :  cris.py
What is it   :  Routines to plot maps for CrIS
Includes     :  plot_daily_overview()
Author        : Frank Werner
Date          : 20240724
Modf          : 20241205: removed species keyword from plot_daily_overview(), replaced mostly with l2standard.species.
                          Also added color_map keyword to plot_daily_overview().
                20250408: adjusted titles, ranges, and colormaps for each species in plot_daily_overview().
                          Also removed color_map keyword.
                20250606: adjusted figure orders and colormaps in plot_daily_overview()
                20250609: added cmap_conc, cmap_error, cmap_dof keywords to plot_daily_overview()
                20250701: species-specific adjustments in plot_daily_overview() following feedback from TROPESS group
                20250924: species-specific adjustments in plot_daily_overview() following more feedback from TROPESS group
                20251023: adjustments to lognorm and cbar_ticks

                               
"""
# Import modules
# =======================================
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as cgeo

from tropessplots.calc.grid_average import grid_average
from tropessplots.plotting.parula_cmap import parula_cmap
from tropessplots.plotting.maps import map_data_1d


# Functions and classes
# =======================================
def plot_daily_overview(l2summary=None,
                        l2standard=None,
                        style='grid_boxes',
                        cmap_conc=parula_cmap(),
                        cmap_error=matplotlib.cm.Blues_r,
                        cmap_dof=matplotlib.cm.magma,#matplotlib.cm.Oranges_r,
                        file_out=''):
    ##############
    # Plots an overview from daily summary/standard files.
    #
    # Parameters
    # ---------
    # l2summary: object; L2 TROPESS summary data
    # l2standard: object; L2 TROPESS standard data
    # style: string; either 'dots' for each indiviudal loaction, 'interp', or 'grid_boxes'
    # cmap_conc, cmap_error, cmap_dof: cmap objects; color maps for concentrations, errors, and DoFs
    # file_out: string; total path to the output file for the figure
    #
    # Returns
    # -------
    # None
    ##############

    # Some housekeeping
    if l2summary is None and l2standard is None:
        raise ValueError("No input data provided")

    # Figure dimensions
    fig = plt.figure(figsize=(12, 14))

    # Statistics
    if l2standard.species == 'CO':
        var = l2summary.col
        unit = ' mol m$^{-2}$'
    if l2standard.species == 'NH3':
        var = l2summary.col
        unit = 'x10$^{15}$ molecules cm$^{-2}$'
    if l2standard.species == 'O3':
        var = l2summary.col_t*2241.1475
        unit = ' DU (Trop.)'
    if l2standard.species == 'CH4':
        var = l2summary.x_col_p/1e3
        unit = ' ppmv (Partial Troposph. Column)'
    if l2standard.species == 'PAN':
        var = l2summary.x_col_ft
        unit = ' ppbv (Free Trop.)'
    if l2standard.species == 'TATM':
        var = l2standard.x[:, 11]
        unit = ' K (at 121.2 hPa)'
    if l2standard.species == 'H2O':
        var = l2standard.x[:, 4]*1e6
        unit = ' ppmv (at 749.9 hPa)'
    if l2standard.species == 'HDO':
        var = l2standard.x[:, 7]*3.11e4
        unit = ' (HDO/H$_2$O/3.11e$^{-4}$ at 510.9 hPa)'
    ind_plot = np.where(var > -900)[0]
    #n_tot = len(var[ind_plot])
    n_good = len(ind_plot)
    min_val = np.min(var[ind_plot])
    max_val = np.max(var[ind_plot])
    
    # Title
    add_text_string_simple = l2standard.species+' ' +\
        str(l2standard.date[0])[0:4]+'-' +\
        str(l2standard.date[0])[4:6]+'-' +\
        str(l2standard.date[0])[6:8]
    if l2standard.date[0] != l2standard.date[-1]:
        add_text_string_simple += ' - ' + str(l2standard.date[-1])[0:4]+'-' +\
        str(l2standard.date[-1])[4:6]+'-' +\
        str(l2standard.date[-1])[6:8]
    
    fig.suptitle(add_text_string_simple, fontsize=10, fontweight='bold', y=0.96)

    # Summary
    if l2summary is None:
        if l2standard.date[0] != l2standard.date[-1]:
            add_text_string = 'CrIS-JPSS-1: '+l2standard.species+' ' +\
                str(l2standard.date[0])[0:4]+'-' +\
                str(l2standard.date[0])[4:6]+'-' +\
                str(l2standard.date[0])[6:8] + ' - ' +\
                str(l2standard.date[-1])[0:4]+'-' +\
                str(l2standard.date[-1])[4:6]+'-' +\
                str(l2standard.date[-1])[6:8] +\
                ', # Retrieved = '+str(n_good) +\
                ', # Min. Val. = '+str(np.round(min_val, 1))+unit +\
                ', # Max. Val. = '+str(np.round(max_val, 1))+unit+'\n' +\
                'Standard File DOI: '+l2standard.global_attrs[0]['IdentifierProductDOI']+'\n'
        else:
            add_text_string = 'CrIS-JPSS-1: '+l2standard.species+' ' +\
                str(l2standard.date[0])[0:4]+'-' +\
                str(l2standard.date[0])[4:6]+'-' +\
                str(l2standard.date[0])[6:8] +\
                ', # Retrieved = '+str(n_good) +\
                ', # Min. Val. = '+str(np.round(min_val, 1))+unit +\
                ', # Max. Val. = '+str(np.round(max_val, 1))+unit+'\n' +\
                'Standard File: '+l2standard.global_attrs[0]['GranuleID'] +\
                ', DOI: '+l2standard.global_attrs[0]['IdentifierProductDOI']+'\n'
    else:
        if l2standard.date[0] != l2standard.date[-1]:
            add_text_string = 'CrIS-JPSS-1: '+l2standard.species+' ' +\
                str(l2standard.date[0])[0:4]+'-' +\
                str(l2standard.date[0])[4:6]+'-' +\
                str(l2standard.date[0])[6:8] + ' - ' +\
                str(l2standard.date[-1])[0:4]+'-' +\
                str(l2standard.date[-1])[4:6]+'-' +\
                str(l2standard.date[-1])[6:8] +\
                ', # Retrieved = '+str(n_good) +\
                ', # Min. Val. = '+str(np.round(min_val, 1))+unit +\
                ', # Max. Val. = '+str(np.round(max_val, 1))+unit+'\n' +\
                'Standard File DOI: '+l2standard.global_attrs[0]['IdentifierProductDOI']+'\n' +\
                'Summary File DOI: '+l2summary.global_attrs[0]['IdentifierProductDOI']
        else:
            add_text_string = 'CrIS-JPSS-1: '+l2standard.species+' ' +\
                str(l2standard.date[0])[0:4]+'-' +\
                str(l2standard.date[0])[4:6]+'-' +\
                str(l2standard.date[0])[6:8] +\
                ', # Retrieved = '+str(n_good) +\
                ', # Min. Val. = '+str(np.round(min_val, 1))+unit +\
                ', # Max. Val. = '+str(np.round(max_val, 1))+unit+'\n' +\
                'Standard File: '+l2standard.global_attrs[0]['GranuleID'] +\
                ', DOI: '+l2standard.global_attrs[0]['IdentifierProductDOI']+'\n' +\
                'Summary File: '+l2summary.global_attrs[0]['GranuleID'] +\
                ', DOI: '+l2summary.global_attrs[0]['IdentifierProductDOI']

    plt.figtext(0.5, 0.01, add_text_string, ha="center", fontsize=10, fontweight='bold')

    for i_plot in range(0, 6):

        # Do map plot
        do_map_plot = True
        
        # Set log_norm
        log_norm = False

        # Species-specific settings
        if l2standard.species == 'CO':
            if i_plot == 0:
                var = l2standard.x[:, 6]*1e+9
                boundaries = np.linspace(40, 200, 256)
                cbar_label = 'CO Volume Mixing Ratio at 215.4 hPa / ppbv'
                cbar_ticks = [40, 60, 80, 100, 120, 140, 160, 180, 200]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 2:
                var = l2standard.x[:, 4]*1e+9
                boundaries = np.linspace(40, 200, 256)
                cbar_label = 'CO Volume Mixing Ratio at 383.2 hPa / ppbv'
                cbar_ticks = [40, 60, 80, 100, 120, 140, 160, 180, 200]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 4:
                var = l2standard.x[:, 2]*1e+9
                boundaries = np.linspace(40, 200, 256)
                cbar_label = 'CO Volume Mixing Ratio at 681.3 hPa / ppbv'
                cbar_ticks = [40, 60, 80, 100, 120, 140, 160, 180, 200]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 1:
                var = l2summary.col
                boundaries = np.linspace(0.5, 3.5, 256)*1e18/(6.022141e+19)
                cbar_label = 'CO Total Column Density / mol m$^{-2}$'
                cbar_ticks = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])*1e18/(6.022141e+19)
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
            if i_plot == 3:
                var = 100*l2summary.col_error/l2summary.col
                boundaries = np.linspace(0, 10, 256)
                cbar_label = 'CO Total Column Error / %'
                cbar_ticks = [0, 2, 4, 6, 8, 10]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = matplotlib.cm.YlGn_r
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(0.2, 1.6, 256)
                cbar_label = 'CO DOFS / 1'
                cbar_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
        if l2standard.species == 'NH3':
            if i_plot == 0:
                var = l2standard.x[:, 5]*1e+9
                ind = np.where(l2standard.land_flag == 0)[0]
                var[ind] = np.nan
                boundaries = np.linspace(0.2, 2, 256)
                cbar_label = 'NH$_3$ Volume Mixing Ratio at 681.3 hPa / ppbv'
                cbar_ticks = [0.2, 0.6, 1, 1.4, 1.8]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = False
            if i_plot == 2:
                var = l2standard.x[:, 4]*1e+9
                ind = np.where(l2standard.land_flag == 0)[0]
                var[ind] = np.nan
                boundaries = np.linspace(0.2, 2, 256)
                cbar_label = 'NH$_3$ Volume Mixing Ratio at 749.9 hPa / ppbv'
                cbar_ticks = [0.2, 0.6, 1, 1.4, 1.8]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = False
            if i_plot == 4:
                var = l2standard.x[:, 2]*1e+9
                ind = np.where(l2standard.land_flag == 0)[0]
                var[ind] = np.nan
                boundaries = np.linspace(1, 10, 256)
                cbar_label = 'NH$_3$ Volume Mixing Ratio at 908.5 hPa / ppbv'
                cbar_ticks = [2, 4, 6, 8, 10]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = False
            if i_plot == 1:
                var = l2summary.col*6.022141e+19/1e15
                ind = np.where(l2standard.land_flag == 0)[0]
                var[ind] = np.nan
                boundaries = np.linspace(1., 20, 256)
                cbar_label = 'NH$_3$ Total Column Density / 10$^{15}$ molecules cm$^{-2}$'
                cbar_ticks = [2, 6, 10, 14, 18]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
                log_norm = False
            if i_plot == 3:
                var = 100*l2summary.col_error/l2summary.col
                ind = np.where(l2standard.land_flag == 0)[0]
                var[ind] = np.nan
                boundaries = np.linspace(0, 80, 256)
                cbar_label = 'NH$_3$ Total Column Error / %'
                cbar_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = matplotlib.cm.YlGn_r
            if i_plot == 5:
                var = l2standard.signal_dof
                ind = np.where(l2standard.land_flag == 0)[0]
                var[ind] = np.nan
                boundaries = np.linspace(0., 1., 256)
                cbar_label = 'NH$_3$ DOFS / 1'
                cbar_ticks = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
        if l2standard.species == 'O3':
            if i_plot == 1:
                var = l2summary.col*2241.1475
                boundaries = np.linspace(270, 370, 256)
                cbar_label = 'O$_3$ Total Column / DU'
                cbar_ticks = [270, 290, 310, 330, 350, 370]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
            if i_plot == 2:
                var = l2summary.col_t*2241.1475
                boundaries = np.linspace(10, 60, 256)
                cbar_label = 'O$_3$ Tropospheric Column / DU'
                cbar_ticks = [10, 20, 30, 40, 50, 60]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
            if i_plot == 4:
                var = l2summary.col_ut*2241.1475
                boundaries = np.linspace(10, 50, 256)
                cbar_label = 'O$_3$ Upper Tropospheric Column / DU'
                cbar_ticks = [10, 20, 30, 40, 50]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
            if i_plot == 0:
                var = l2standard.x[:, 5]*1e+9
                boundaries = np.linspace(20, 100, 256)
                cbar_label = 'O$_3$ Volume Mixing Ratio at 464.2 hPa / ppbv'
                cbar_ticks = [20, 30, 40, 50, 60, 70, 80, 90, 100]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 3:
                var = 100*l2summary.col_error/l2summary.col
                boundaries = np.linspace(0, 5, 256)
                cbar_label = 'O$_3$ Total Column Error / %'
                cbar_ticks = [0, 1, 2, 3, 4, 5]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = matplotlib.cm.YlGn_r
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(2., 5., 256)
                cbar_label = 'O$_3$ DOFS / 1'
                cbar_ticks = [2, 3, 4, 5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
        if l2standard.species == 'CH4':
            if i_plot == 0:
                var = l2standard.x[:, 9]*1e+6
                boundaries = np.linspace(1.9, 2.2, 256)
                cbar_label = 'CH$_4$ Volume Mixing Ratio at 215.4 hPa / ppmv'
                cbar_ticks = [1.9, 2.0, 2.1, 2.2]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 2:
                var = l2standard.x[:, 5]*1e+6
                boundaries = np.linspace(1.9, 2.2, 256)
                cbar_label = 'CH$_4$ Volume Mixing Ratio at 464.2 hPa / ppmv'
                cbar_ticks = [1.9, 2.0, 2.1, 2.2]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 4:
                var = l2standard.x[:, 3]*1e+6
                boundaries = np.linspace(1.9, 2.2, 256)
                cbar_label = 'CH$_4$ Volume Mixing Ratio at 681.3 hPa / ppmv'
                cbar_ticks = [1.9, 2.0, 2.1, 2.2]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 1:
                var = l2summary.x_col_p/1e3
                boundaries = np.linspace(1.9, 2.2, 256)
                cbar_label = 'CH$_4$ Partial Tropospheric Column Density / ppmv'
                cbar_ticks = [1.9, 2.0, 2.1, 2.2]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
            if i_plot == 3:
                var = 100*l2summary.x_col_p_error/l2summary.x_col_p
                boundaries = np.linspace(0.3, 0.9, 256)
                cbar_label = 'CH$_4$ Partial Tropospheric Column Error / %'
                cbar_ticks = [0.3, 0.5, 0.7, 0.9]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = matplotlib.cm.YlGn_r
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(0.8, 2.4, 256)
                cbar_label = 'CH$_4$ DOFS / 1'
                cbar_ticks = [0.8, 1.2, 1.6, 2.0, 2.4]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
        if l2standard.species == 'PAN':
            if i_plot == 1:
                var = l2summary.x_col_ft
                boundaries = np.linspace(0., 0.8, 256)
                cbar_label = 'PAN Volume Mixing Ratio in Free Troposphere / ppbv'
                cbar_ticks = [0, 0.2, 0.4, 0.6, 0.8]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = cmap_conc
            if i_plot == 0:
                var = l2standard.x[:, 5]*1e+9
                boundaries = np.linspace(0, 0.5, 256)
                cbar_label = 'PAN Volume Mixing Ratio at 464.2 hPa / ppbv'
                cbar_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                do_map_plot = False
            if i_plot == 3:
                var = 100*l2summary.x_col_ft_error/l2summary.x_col_ft
                boundaries = np.linspace(0, 40, 256)
                cbar_label = 'PAN Error in Free Troposphere / %'
                cbar_ticks = [0, 10, 20, 30, 40]
                lon = l2summary.longitude
                lat = l2summary.latitude
                cmap = matplotlib.cm.YlGn_r
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(0.1, 1.1, 256)
                cbar_label = 'PAN DOFS / 1'
                cbar_ticks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
            if i_plot == 2:
                var = l2standard.x[:, 3]*1e+9
                boundaries = np.linspace(0, 0.5, 256)
                cbar_label = 'PAN Volume Mixing Ratio at 681.3 hPa / ppbv'
                cbar_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                do_map_plot = False
            if i_plot == 4:
                var = l2standard.x[:, 2]*1e+9
                boundaries = np.linspace(0, 0.5, 256)
                cbar_label = 'PAN Volume Mixing Ratio at 825.4 hPa / ppbv'
                cbar_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                do_map_plot = False
        if l2standard.species == 'TATM':
            if i_plot == 0:
                var = l2standard.x[:, 28]
                boundaries = np.linspace(230, 285, 256)
                cbar_label = 'Atmospheric Temperature at 1 hPa / K'
                cbar_ticks = [230, 240, 250, 260, 270, 280]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 1:
                var = l2standard.x[:, 22]
                boundaries = np.linspace(190, 245, 256)
                cbar_label = 'Atmospheric Temperature at 10 hPa / K'
                cbar_ticks = [190, 200, 210, 220, 230, 240]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 2:
                var = l2standard.x[:, 14]
                boundaries = np.linspace(190, 240, 256)
                cbar_label = 'Atmospheric Temperature at 100 hPa / K'
                cbar_ticks = [190, 200, 210, 220, 230, 240]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 3:
                var = l2standard.x[:, 8]
                boundaries = np.linspace(225, 270, 256)
                cbar_label = 'Atmospheric Temperature at 421.7 hPa / K'
                cbar_ticks = [225, 235, 245, 255, 265]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 4:
                var = l2standard.x[:, 4]
                boundaries = np.linspace(240, 300, 256)
                cbar_label = 'Atmospheric Temperature at 749.9 hPa / K'
                cbar_ticks = [240, 250, 260, 270, 280, 290, 300]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(6., 10., 256)
                cbar_label = 'Atmospheric Temperature DOFS / 1'
                cbar_ticks = [6, 7, 8, 9, 10]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
        if l2standard.species == 'H2O':
            if i_plot == 0:
                var = l2standard.x[:, 16]*1e+6
                boundaries = np.geomspace(5, 9, 256)
                cbar_label = 'H$_2$O Volume Mixing Ratio at 0.1 hPa / ppmv'
                cbar_ticks = [5, 6, 7, 8, 9]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = True
            if i_plot == 1:
                var = l2standard.x[:, 14]*1e+6
                boundaries = np.geomspace(1, 4.5, 256)
                cbar_label = 'H$_2$O Volume Mixing Ratio at 75.0 hPa / ppmv'
                cbar_ticks = [1.2, 2, 3, 4.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = True
            if i_plot == 2:
                var = l2standard.x[:, 13]*1e+6
                boundaries = np.geomspace(3, 18, 256)
                cbar_label = 'H$_2$O Volume Mixing Ratio at 133.4 hPa / ppmv'
                cbar_ticks = [3, 6, 9, 12, 15]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = True
            if i_plot == 3:
                var = l2standard.x[:, 8]*1e+6
                boundaries = np.geomspace(30, 6000, 256)
                cbar_label = 'H$_2$O Volume Mixing Ratio at 421.7 hPa / ppmv'
                cbar_ticks = [50, 100, 200, 500, 1000, 2000, 5000]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = True
            if i_plot == 4:
                var = l2standard.x[:, 4]*1e+6
                boundaries = np.geomspace(100, 40000, 256)
                cbar_label = 'H$_2$O Volume Mixing Ratio at 749.9 hPa / ppmv'
                cbar_ticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                log_norm = True
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(1., 6., 256)
                cbar_label = 'H$_2$O DOFS / 1'
                cbar_ticks = [1, 2, 3, 4, 5, 6]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
        if l2standard.species == 'HDO':
            if i_plot == 0:
                var = l2standard.x[:, 8]*3.11e4
                boundaries = np.linspace(5.5, 10, 256)
                cbar_label = 'HDO/H$_2$O/3.11e$^{-4}$ at 421.7 hPa / 1'
                cbar_ticks = [5.5, 6.5, 7.5, 8.5, 9.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                do_map_plot = False
            if i_plot == 1:
                var = l2standard.x[:, 7]*3.11e4
                boundaries = np.linspace(5.5, 10, 256)
                cbar_label = 'HDO/H$_2$O/3.11e$^{-4}$ at 510.9 hPa / 1'
                cbar_ticks = [5.5, 6.5, 7.5, 8.5, 9.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 2:
                var = l2standard.x[:, 6]*3.11e4
                boundaries = np.linspace(5.5, 10, 256)
                cbar_label = 'HDO/H$_2$O/3.11e$^{-4}$ at 618.5 hPa / 1'
                cbar_ticks = [5.5, 6.5, 7.5, 8.5, 9.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                do_map_plot = False
            if i_plot == 3:
                var = l2standard.x[:, 5]*3.11e4
                boundaries = np.linspace(5.5, 10, 256)
                cbar_label = 'HDO/H$_2$O/3.11e$^{-4}$ at 681.1 hPa / 1'
                cbar_ticks = [5.5, 6.5, 7.5, 8.5, 9.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
            if i_plot == 4:
                var = l2standard.x[:, 5]*3.11e4
                boundaries = np.linspace(5.5, 10, 256)
                cbar_label = 'HDO/H$_2$O/3.11e$^{-4}$ at 681.1 hPa / 1'
                cbar_ticks = [5.5, 6.5, 7.5, 8.5, 9.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = cmap_conc
                do_map_plot = False
            if i_plot == 5:
                var = l2standard.signal_dof
                boundaries = np.linspace(0.1, 2.5, 256)
                log_norm=True
                cbar_label = 'HDO DOFS / 1'
                cbar_ticks = [0.1, 0.5, 1, 1.5, 2, 2.5]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma
                log_norm = True

        if do_map_plot is True:
            
            # Filter data
            ind_plot = np.where(var > -900)[0]

            # Gridding
            if style == 'grid_boxes':
                var_dummy, lon_edges, lat_edges = grid_average(x=lon[ind_plot],
                                                               y=lat[ind_plot],
                                                               values=var[ind_plot],
                                                               x_bins=np.arange(-180,
                                                                                181, 1),
                                                               y_bins=np.arange(-90,
                                                                                91, 1),
                                                               statistic='mean'
                                                               )
                var = var_dummy
                lon = lon_edges[0:-1] + 0.5*(lon_edges[1]-lon_edges[0])
                lat = lat_edges[0:-1] + 0.5*(lat_edges[1]-lat_edges[0])

            if i_plot % 2 == 0:
                left_labels = True
                right_labels = False
            if i_plot % 2 == 1:
                left_labels = False
                right_labels = True

            # Define axis
            ax = plt.subplot(3, 2, i_plot+1, projection=ccrs.PlateCarree())

            # Plot map
            if style != 'grid_boxes':
                map_data_1d(ax=ax,
                            lon=lon[ind_plot],
                            lat=lat[ind_plot],
                            var=var[ind_plot],
                            mask=None,
                            cmap=cmap,
                            boundaries=boundaries,
                            log_norm=log_norm,
                            style=style,
                            borders=False,
                            markersize=0.5,
                            extent=[-180, 180, -90, 90],
                            left_labels=left_labels,
                            right_labels=right_labels,
                            xlocator=[-120, -60, 0, 60, 120],
                            ylocator=[-90, -60, -30, 0, 30, 60, 90],
                            cbar_pad=0.08,
                            cbar_ticks=cbar_ticks,
                            #cbar_no_minorticks=True,
                            cbar_minor_nbins=2,
                            cbar_label=cbar_label,
                            cbar_fs=10,
                            add_text=None,
                            fig_title='',
                            fig_title_pad=41
                            )
            else:
                map_data_1d(ax=ax,
                            lon=lon,
                            lat=lat,
                            var=var.T,
                            mask=None,
                            cmap=cmap,
                            boundaries=boundaries,
                            log_norm=log_norm,
                            style=style,
                            borders=False,
                            markersize=0.5,
                            extent=[-180, 180, -90, 90],
                            left_labels=left_labels,
                            right_labels=right_labels,
                            xlocator=[-120, -60, 0, 60, 120],
                            ylocator=[-90, -60, -30, 0, 30, 60, 90],
                            cbar_fraction=0.288,
                            cbar_pad=0.08,
                            cbar_aspect=35,
                            cbar_ticks=cbar_ticks,
                            #cbar_no_minorticks=True,
                            cbar_minor_nbins=2,
                            cbar_label=cbar_label,
                            cbar_fs=10,
                            add_text=None,
                            fig_title='',
                            fig_title_pad=41
                            )

    plt.tight_layout(h_pad=-4.95, w_pad=0.35)
    
    # Center the PAN plot
    if l2standard.species == 'PAN' or l2standard.species == 'HDO':
        fig.canvas.draw()  # Freeze positions, then shift
        SHIFT = 0.2  # tweak to taste (fraction of figure width)
        for ax in fig.axes:
            is_cbar = ax.get_label() == '<colorbar>'
            if isinstance(ax, cgeo.GeoAxes) or is_cbar:
                bb = ax.get_position()
                ax.set_position([max(0.02, bb.x0 - SHIFT), bb.y0, bb.width, bb.height])

    if file_out == '':
        plt.show()
    else:
        fig.savefig(file_out,
                    format='png',
                    dpi=150,
                    bbox_inches='tight')
    plt.close()

    # Return features
    return(None)
