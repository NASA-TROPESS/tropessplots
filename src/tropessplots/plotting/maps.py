"""
Title        :  maps.py
What is it   :  Routines to plot cartopy maps
Includes     :  map_data_1d()
Author        : Frank Werner
Date          : 20240722
Modf          : 20250124 - added pixels keyqord to map_data_1d()
                20250213 - fixed ax.add_feature(cfeature.BORDERS) call in
                           map_data_1d() by removing resolution keyword
                20250214 - added borders keyword to map_data_1d()
                20250217 - added log_norm keyword to map_data_1d()
                20250220 - changed 'pixels' to 'polygons' for style keyword in 
                           map_data_1d(), added 'ellipse' option
                20250331 - added projection keyword to map_data_1d()
                20250606 - changed formatting of ax.coastlines() and ax.add_feature() in map_data_1d()
                20250924 - fixed colorbar tickmarks for log_norm=True in map_data_1d()
                20251023 - NaN-safe LogNorm and robust vmin/vmax; force-exact cbar ticks; 
                           scientific tick labels for log colorbars only; respect user boundaries in log mode
                20251030 - create the colorbar from the figure and tell it which Axes to use in map_data_1d(),
                           updated _force_colorbar_ticks to work with newer matplotlib versions
"""

# Import modules
# =======================================
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import (
    AutoMinorLocator, FixedLocator, NullLocator, FuncFormatter, LogLocator, ScalarFormatter
)

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

from tropessplots.plotting.parula_cmap import parula_cmap

import warnings
warnings.filterwarnings('ignore')


# ---- Helpers (internal) ------------------------------------------------------

def _sci_label(x):
    """
    Format a positive number x in scientific form as 'a × 10^b' using mathtext,
    e.g., 5   -> '$5\\times10^{0}$'
          0.5 -> '$5\\times10^{-1}$'
          12  -> '$1.2\\times10^{1}$'
    """
    if not np.isfinite(x) or x <= 0:
        return ""
    exp = int(np.floor(np.log10(x)))
    mant = x / (10 ** exp)
    return rf"${mant:g}\times10^{{{exp}}}$"


def _force_colorbar_ticks(cbar, ticks, *, log_mode=False, scientific=False):
    """
    Force a colorbar to show exactly the given tick positions.
    - Filters to valid range: >0 for log mode, and within vmin..vmax.
    - Sets both the axis locator and the Colorbar ticks.
    - Optionally formats labels in scientific form for log colorbars.
    """
    vmin, vmax = float(cbar.norm.vmin), float(cbar.norm.vmax)

    if log_mode:
        ticks = [t for t in ticks if (t is not None) and np.isfinite(t) and (t > 0) and (vmin <= t <= vmax)]
    else:
        ticks = [t for t in ticks if (t is not None) and np.isfinite(t) and (vmin <= t <= vmax)]

    axis = cbar.ax.xaxis if cbar.orientation == "horizontal" else cbar.ax.yaxis

    # Exact positions via locator; no minors by default
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())

    # Version-safe: don't pass update_ticks kwarg
    cbar.set_ticks(ticks)

    # Labels
    if scientific and log_mode:
        cbar.set_ticklabels([_sci_label(t) for t in ticks])
    else:
        cbar.set_ticklabels([f"{t:g}" for t in ticks])

    return ticks


def _robust_range(var, mask=None, *, log_mode=False, p_lo=1, p_hi=99):
    """
    Compute robust vmin/vmax from var ignoring NaNs and masked points.
    If log_mode=True, also drop non-positive values and ensure vmin>0.
    Falls back to nanmin/nanmax if percentiles are ill-defined.
    Raises ValueError if no valid samples remain.
    """
    v = np.asarray(var, dtype=float)
    valid = np.isfinite(v)
    if mask is not None:
        valid &= (np.asarray(mask) == 0)
    if log_mode:
        valid &= (v > 0)

    if not np.any(valid):
        raise ValueError("No valid data for color scaling (all NaN/masked"
                         + (", or <=0 for LogNorm" if log_mode else "") + ").")

    vv = v[valid]
    try:
        vmin = float(np.nanpercentile(vv, p_lo))
        vmax = float(np.nanpercentile(vv, p_hi))
    except Exception:
        vmin = float(np.nanmin(vv))
        vmax = float(np.nanmax(vv))

    # guard against equal or inverted bounds
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(vv))
        vmax = float(np.nanmax(vv))
        if vmin == vmax:
            # expand a tiny bit to avoid singular norm
            if log_mode:
                vmin = vmin / 1.01 if vmin > 0 else np.nextafter(0.0, 1.0)
                vmax = max(vmax * 1.01, vmin * 1.01)
            else:
                vmin = vmin - 1e-12
                vmax = vmax + 1e-12

    if log_mode and vmin <= 0:
        vmin = float(np.nanmin(vv[vv > 0]))

    return vmin, vmax


# Functions
# =======================================
def map_data_1d(ax=None,
                projection='PlateCarree',
                lon=None,
                lat=None,
                lon_pixels=None,
                lat_pixels=None,
                var=None,
                mask=None,
                cmap=None,
                boundaries=None,
                log_norm=False,
                style='dots',
                markersize=1,
                coastlines_res='10m',
                borders=True,
                extent=[-180, 180, -90, 90],
                bottom_labels=True,
                top_labels=True,
                left_labels=True,
                right_labels=True,
                xlocator=[-135, -90, -45, 0, 45, 90, 135],
                ylocator=[-90, -45, 0, 45, 90],
                grid_fs=8,
                cbar_ticks=None,
                cbar_fraction=0.0188,
                cbar_pad=0.08,
                cbar_orientation='horizontal',
                cbar_aspect=96,
                cbar_no_minorticks=False,
                cbar_minor_nbins=None,
                cbar_label='',
                cbar_rotation=0,
                cbar_labelpad=5,
                cbar_fs=12,
                cbar_do=True,
                add_text=None,
                add_text_fs=12,
                fig_title='',
                fig_title_pad=0
                ):
    ##############
    # Plots a map with 1d longitude, latitude, and data arrays.
    #
    # Parameters
    # ---------
    # ax: pyplot axis; the axis into which to plot the map
    # projection: string; map projection
    # lon, lat: ndarrays; longitude and latitude
    # var: ndarray; the variable to plot
    # mask (optional): ndarray; boolean as binary for masking data points
    # cmap (optional): pyplot colormap or string; color map to be used
    # boundaries (optional): ndarray; increments for the colormap
    # log_norm (optional): boolean; whether to use log-normal normalization
    # style (optional): string; 'dots' or 'interp' or 'grid_boxes' or 'polygons' or 'ellipse'
    # markersize (optional): integer or float; size of 'dots'
    # coastlines_res (optional): string; resolution for cartopy coastlines
    # borders (optional): boolean; whether to plot country borders
    # extent (optional): ndarray; extent of the map
    # bottom_labels, top_labels, left_labels, right_labels (optional): boolean;
    #                   whether to plot zonal and meridional labels
    # xlocator, ylocator (optional): ndarrays; locations of labels
    # grid_fs (optional): integer or float; font size of the labels (default for 12, 6)
    # cbar_ticks down to cbar_do (optional): integer, float, strings, and boolean;
    #                   settings for the colorbar (defaults for 12, 6)
    # add_text (optional): ndarray; position and string for additional text
    # add_text_fs (optional): integer or float; font size of add_text (default for 12, 6)
    # fig_title (optional): string; title of the figure
    # fig_title_pad (optional): integer or float; padding for the title of the figure
    #
    # Returns
    # -------
    # None
    ##############

    # Projection
    if projection == 'PlateCarree':
        crs_proj = ccrs.PlateCarree()
    if projection == 'Mollweide':
        crs_proj = ccrs.Mollweide()
    
    # Define mask
    if mask is None:
        mask = np.zeros_like(var)
    ind_plot = np.where(mask == 0)[0]

    # Define colormap
    if cmap is None:
        cmap = matplotlib.cm.Spectral_r
    if cmap == "Parula":
        cmap = parula_cmap()

    # ---- Define normalization & boundaries (NaN-safe) ----
    if boundaries is None:
        # derive range robustly from data (ignoring NaNs/mask; >0 for log)
        vmin, vmax = _robust_range(var, mask=mask, log_mode=log_norm, p_lo=1, p_hi=99)
        if log_norm:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
            boundaries = None  # no discrete boundaries provided by user
        else:
            boundaries = np.linspace(vmin, vmax, 41)
            n_colors = cmap.N if hasattr(cmap, "N") else matplotlib.cm.get_cmap(cmap).N
            norm = matplotlib.colors.BoundaryNorm(boundaries, n_colors, clip=True)
    else:
        # user-supplied boundaries; sanitize
        b = np.asarray(boundaries, dtype=float)
        if log_norm:
            b = b[np.isfinite(b) & (b > 0)]
            if b.size == 0:
                raise ValueError("Log colorbar: boundaries must be positive/finite.")
            vmin, vmax = float(b.min()), float(b.max())
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
            boundaries = b  # keep boundaries for colorbar tick placement
        else:
            b = b[np.isfinite(b)]
            if b.size == 0:
                raise ValueError("Linear colorbar: boundaries must be finite.")
            boundaries = b
            n_colors = cmap.N if hasattr(cmap, "N") else matplotlib.cm.get_cmap(cmap).N
            norm = matplotlib.colors.BoundaryNorm(boundaries, n_colors, clip=True)

    # Clean ScalarMappable for the colorbar (independent of plotted artist)
    sm = ScalarMappable(norm=norm, cmap=(cmap if hasattr(cmap, "N") else matplotlib.cm.get_cmap(cmap)))
    sm.set_array([])

    # Plot dots
    if style == 'dots':
        var2 = np.zeros_like(var)
        var2[:] = np.nan
        plt.tripcolor(lon[ind_plot], lat[ind_plot],
                      var2[ind_plot], cmap=cmap, norm=norm)
        for i in range(0, len(ind_plot)):
            c = cmap(norm(var[i]))
            plt.plot(lon[ind_plot[i]], lat[ind_plot[i]],
                     'o', color=c, markersize=markersize, zorder=0)

    # Plot unstructured triangular grid
    if style == 'interp':
        # Mask NaN/invalids explicitly (esp. important for log)
        if log_norm:
            valid = np.isfinite(var) & (var > 0) & (mask == 0)
        else:
            valid = np.isfinite(var) & (mask == 0)
        plt.tripcolor(lon[valid], lat[valid],
                      var[valid], cmap=cmap, norm=norm)

    # Plot grid boxes
    if style == 'grid_boxes':
        xx, yy = np.meshgrid(lon, lat)
        plt.pcolor(xx, yy, var, cmap=cmap, norm=norm)
    
    # Plot pixels
    if style == 'polygons':
        var2 = np.zeros_like(var)
        var2[:] = np.nan
        plt.tripcolor(lon[ind_plot], lat[ind_plot],
                      var2[ind_plot], cmap=cmap, norm=norm)
        for i in range(0, len(lat_pixels)):
            c = cmap(norm(var[i]))
            plt.fill(lon_pixels[i,:],
                     lat_pixels[i,:],
                     color=c,
                     zorder=0)
    
    # Plot ellipse
    if style == 'ellipse':
        var2 = np.zeros_like(var)
        var2[:] = np.nan
        plt.tripcolor(lon[ind_plot], lat[ind_plot],
                      var2[ind_plot], cmap=cmap, norm=norm)
        for i in range(0, len(lat_pixels)):
            c = cmap(norm(var[i]))
            A = np.stack([
                lon_pixels[i,:]**2,
                lon_pixels[i,:] * lat_pixels[i,:],
                lat_pixels[i,:]**2,
                lon_pixels[i,:],
                lat_pixels[i,:]
            ]).T
        bvec = np.ones_like(lon_pixels[i,:])
        w = np.linalg.lstsq(A, bvec, rcond=None)[0].squeeze()
        xlin = np.linspace(np.min(lon_pixels[i,:]), np.max(lon_pixels[i,:]), 200)
        ylin = np.linspace(np.min(lat_pixels[i,:]), np.max(lat_pixels[i,:]), 200)
        X, Y = np.meshgrid(xlin, ylin)
        Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 + w[3]*X + w[4]*Y
        ax.contourf(X, Y, Z, [1, 2], colors=[c, 'tab:white'])

    # Add coastlines and borders
    ax.coastlines(coastlines_res, color='black', linewidth=1, alpha=0.5)
    if borders is True:
        ax.add_feature(cfeature.BORDERS, color='black', linewidth=1, alpha=0.5)

    # Define the extent
    ax.set_extent(extent)

    # Plot gridlines
    gl = ax.gridlines(crs=crs_proj, draw_labels=True,
                      linewidth=1, color='gray', alpha=0.52, linestyle='--')
    gl.bottom_labels = bottom_labels
    gl.top_labels = top_labels
    gl.left_labels = left_labels
    gl.right_labels = right_labels
    gl.xlocator = matplotlib.ticker.FixedLocator(xlocator)
    gl.ylocator = matplotlib.ticker.FixedLocator(ylocator)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': grid_fs, 'color': 'gray', 'weight': 'bold'}
    gl.ylabel_style = {'size': grid_fs, 'color': 'gray', 'weight': 'bold'}

    # Colorbar
    if cbar_do is True:
        # Decide ticks to pass at creation time
        # - For log: if user gave no ticks but gave boundaries, use boundaries as ticks
        # - For linear: pass through user ticks
        if log_norm:
            if cbar_ticks is None and boundaries is not None:
                cbar_ticks_eff = list(boundaries)
            else:
                cbar_ticks_eff = None  # we'll force later or auto
        else:
            cbar_ticks_eff = cbar_ticks

        cbar = plt.colorbar(
            sm,
            ax=ax,  # <-- crucial: tells Matplotlib where to steal space
            ticks=cbar_ticks_eff,
            fraction=cbar_fraction,
            pad=cbar_pad,
            orientation=cbar_orientation,
            aspect=cbar_aspect,
            boundaries=boundaries,  # keep boundaries for both linear & log
        )

        cbar.ax.tick_params(labelsize=cbar_fs)
        if cbar_no_minorticks is True:
            cbar.minorticks_off()
        if cbar_minor_nbins is not None and cbar_orientation == 'horizontal':
            cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(n=cbar_minor_nbins))

        if log_norm:
            axis = cbar.ax.xaxis if cbar_orientation == 'horizontal' else cbar.ax.yaxis

            if cbar_ticks is not None:
                # User-provided ticks: force exactly these, scientific labels
                _ = _force_colorbar_ticks(cbar, cbar_ticks, log_mode=True, scientific=True)
            elif boundaries is not None:
                # Boundaries-as-ticks: force and format scientifically
                _ = _force_colorbar_ticks(cbar, list(boundaries), log_mode=True, scientific=True)
            else:
                # True auto (no boundaries, no ticks): locator + scientific labels
                axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
                axis.set_minor_locator(NullLocator())
                axis.set_major_formatter(FuncFormatter(lambda x, pos: _sci_label(x)))

            cbar.ax.tick_params(which='both', labelsize=cbar_fs)

        else:
            # Linear colorbar: keep plain numeric labels explicitly
            axis = cbar.ax.xaxis if cbar_orientation == 'horizontal' else cbar.ax.yaxis
            axis.set_major_formatter(ScalarFormatter(useMathText=False, useOffset=False))

        cbar.set_label(cbar_label,
                       rotation=cbar_rotation,
                       labelpad=cbar_labelpad,
                       fontsize=cbar_fs)
    
    # Additional text
    if add_text is not None:
        plt.text(add_text[0], 
                 add_text[1], 
                 add_text[2], 
                 fontsize=add_text_fs)
    
    if fig_title != '':
        plt.title(fig_title, 
                  fontsize=cbar_fs, 
                  pad=fig_title_pad)

    # Return data
    return(None)
