"""
Title        :  airs_omi.py
What is it   :  Routines to plot maps for AIRS_OMI
Includes     :  plot_daily_overview()
Author        : Frank Werner
Date          : 20251008
Modf          : 20xxxxxx: NaN

                               
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
    if l2standard.species == 'O3':
        var = l2summary.col_t*2241.1475
        unit = ' DU (Trop.)'
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
            add_text_string = 'AIRS-Aqua_OMI-Aura: '+l2standard.species+' ' +\
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
            add_text_string = 'AIRS-Aqua_OMI-Aura: '+l2standard.species+' ' +\
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
            add_text_string = 'AIRS-Aqua_OMI-Aura: '+l2standard.species+' ' +\
                str(l2standard.date[0])[0:4]+'-' +\
                str(l2standard.date[0])[4:6]+'-' +\
                str(l2standard.date[0])[6:8]+ ' - ' +\
                str(l2standard.date[-1])[0:4]+'-' +\
                str(l2standard.date[-1])[4:6]+'-' +\
                str(l2standard.date[-1])[6:8] +\
                ', # Retrieved = '+str(n_good) +\
                ', # Min. Val. = '+str(np.round(min_val, 1))+unit +\
                ', # Max. Val. = '+str(np.round(max_val, 1))+unit+'\n' +\
                'Standard File DOI: '+l2standard.global_attrs[0]['IdentifierProductDOI']+'\n' +\
                'Summary File DOI: '+l2summary.global_attrs[0]['IdentifierProductDOI']
        else:
            add_text_string = 'AIRS-Aqua_OMI-Aura: '+l2standard.species+' ' +\
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
                boundaries = np.linspace(4., 7., 256)
                cbar_label = 'O$_3$ DOFS / 1'
                cbar_ticks = [4, 5, 6, 7]
                lon = l2standard.longitude
                lat = l2standard.latitude
                cmap = matplotlib.cm.magma

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
