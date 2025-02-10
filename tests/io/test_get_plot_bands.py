# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

# =============================================================================
# IMPORTS
# =============================================================================

from Pynomic.core import core
from Pynomic.io import get_plot_bands
from Pynomic.io.get_plot_bands import read_zarr

import pandas as pd
import geopandas as gpd

import shutil

import pytest

import zarr


# =============================================================================
# FUNCTIONS
# =============================================================================


def test_proces_stack_tiff():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    assert isinstance(pyt, core.Pynomicproject)

    assert isinstance(pyt.bands_name, list)
    #assert isinstance(pyt.raw_data, zarr.Group)
    assert isinstance(pyt.ldata, gpd.GeoDataFrame)
    assert isinstance(pyt.dates, list)
    assert isinstance(pyt.n_bands, int)
    assert isinstance(pyt.n_dates, int)

    assert pyt.n_bands == 3
    assert pyt.bands_name[0] == "red"
    assert pyt.bands_name[1] == "green"
    assert pyt.bands_name[2] == "blue"

    return


def test_proces_stack_tiff_no_name_bands():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights", "add_on/Grids/Labmert_test_grid.geojson", "fid"
    )

    assert isinstance(pyt, core.Pynomicproject)
    assert isinstance(pyt.bands_name, list)
    #assert isinstance(pyt.raw_data, zarr.Group)
    assert isinstance(pyt.ldata, gpd.GeoDataFrame)
    assert isinstance(pyt.dates, list)
    assert isinstance(pyt.n_bands, int)
    assert isinstance(pyt.n_dates, int)

    assert pyt.n_bands == 3
    assert pyt.bands_name[0] == "band_1"
    assert pyt.bands_name[1] == "band_2"
    assert pyt.bands_name[2] == "band_3"

    return


#def test_grid_reader_error():
#
#    with pytest.raises(ValueError):
#        get_plot_bands.process_stack_tiff(
#            "add_on/flights", "add_on/Grids/Labmert_test_grid.shp", "fid"
#        )

#    return


def test_read_zarr():
    shutil.rmtree('add_on/zarr_data/RGB_group.zarr', ignore_errors=False)
    tyt = get_plot_bands.process_stack_tiff(
        "add_on/flights", "add_on/Grids/Labmert_test_grid.geojson", "fid"
    )

    tyt.save("add_on/zarr_data/RGB_group.zarr")
    pyt = read_zarr("add_on/zarr_data/RGB_group.zarr")
    assert isinstance(pyt, core.Pynomicproject)
    assert isinstance(pyt.bands_name, list)
    assert isinstance(pyt.raw_data, zarr.hierarchy.Group)
    assert isinstance(pyt.ldata, pd.DataFrame)
    assert isinstance(pyt.dates, list)
    assert isinstance(pyt.n_bands, int)
    assert isinstance(pyt.n_dates, int)

    assert pyt.n_bands == 3
    assert pyt.bands_name[0] == "band_1"
    assert pyt.bands_name[1] == "band_2"
    assert pyt.bands_name[2] == "band_3"

    shutil.rmtree('add_on/zarr_data/RGB_group.zarr', ignore_errors=False)
    return
