# TROPESS Website Plots (tropessplots)

[![Language](https://img.shields.io/badge/python-3.12-blue)](#)

The `tropessplots` library in this repository contains notebooks in the `notebooks` directory that lets you create your own TROPESS daily and multi-day plots, as seen on the TROPESS website at [https://tes.jpl.nasa.gov/tropess/get-data/plots](https://tes.jpl.nasa.gov/tropess/get-data/plots).

Right now, there are four Jupyter notebooks:

- `01_cris_download_run.ipynb`:  Create a daily TROPESS CrIS-JPSS1 plot for all available species.
- `02_airs-omi_download_runipynb`:  Create a daily TROPESS AIRS-OMI ozone plot.
- `03_cris_download_run_range.ipynb`:  Create a multi-day TROPESS CrIS-JPSS1 plot from a range of dates for all available species.
- `04_airs-omi_download_run_range.ipynb`:  Create a multi-day TROPESS AIRS-OMI ozone plot.

These notebooks (and accompanying code) will allow you to tailor the plots to your needs, for example, if you want to change the colorbars.

## Installation

To install necessary libraries and functions to run the plots in a Python 3.12 (or greater) enviornment, you will need to `pip install` this repository like:

```bash
 pip install git+https://github.com/NASA-TROPESS/tropessplots.git
 ```

 Your notebooks will then be able to call various parts of the library like:

 ```python
 from tropessplots.shared.read_nc import read_nc
 ```

 ## Notebooks

 The notebooks are not included in the instllation, you will have to separately clone or download them.