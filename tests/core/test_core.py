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

import numpy as np

import pandas as pd

import pytest

import zarr

# =============================================================================
# FUNCTIONS
# =============================================================================


def test_Pynomicporject_obj():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    assert isinstance(pyt, core.Pynomicproject)

    assert isinstance(pyt.bands_name, list)
    assert isinstance(pyt.raw_data, zarr.hierarchy.Group)
    assert isinstance(pyt.ldata, pd.DataFrame)
    assert isinstance(pyt.dates, list)
    assert isinstance(pyt.n_bands, int)
    assert isinstance(pyt.n_dates, int)

    assert pyt.n_dates == 7
    assert pyt.ldata.shape == (280, 7)
    assert pyt.ldata.columns[0] == "id"
    assert pyt.ldata.columns[1] == "fid"
    assert pyt.ldata.columns[2] == "date"
    assert pyt.ldata.columns[3] == "red"
    assert pyt.ldata.columns[4] == "green"
    assert pyt.ldata.columns[5] == "blue"
    assert pyt.dates[0] == "20180815"
    assert pyt.dates[1] == "20180917"
    assert pyt.dates[2] == "20180905"
    assert pyt.dates[3] == "20180914"
    assert pyt.dates[4] == "20180822"
    assert pyt.n_bands == 3
    assert pyt.bands_name[0] == "red"
    assert pyt.bands_name[1] == "green"
    assert pyt.bands_name[2] == "blue"

    return


def test_getitem():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    assert pyt["n_dates"] == 7
    assert pyt["n_bands"] == 3
    with pytest.raises(KeyError):
        pyt["n_bas"]

    return


def test_generate_unique_feature():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    def VDVI_index(df):
        red = np.mean(df["red"])
        green = np.mean(df["green"])
        blue = np.mean(df["blue"])

        return [(2 * green - red - blue) / (2 * green + red + blue)]

    data = pyt.generate_unique_feature(VDVI_index, ["VDVI"])

    assert isinstance(data, pd.DataFrame)

    assert data.loc[0, "VDVI"] == 0.1546743079777065
    assert data.loc[3, "VDVI"] == 0.1554692596154336
    assert data.loc[275, "VDVI"] == 0.1278593224337763
    assert pyt.ldata.shape[1] == 7

    pyt.generate_unique_feature(VDVI_index, ["VDVI"], to_data=True)
    assert pyt.ldata.shape[1] == 8

    return


def test_senescence_prediction():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    def VDVI_index(df):
        red = np.mean(df["red"])
        green = np.mean(df["green"])
        blue = np.mean(df["blue"])

        return [(2 * green - red - blue) / (2 * green + red + blue)]

    pyt.generate_unique_feature(VDVI_index, ["VDVI"], True)

    df1 = pyt.get_senescens_predictions("VDVI", 0.1)

    assert int(df1.loc[df1.id == 1, "dpred"].values[0]) == 13
    assert int(df1.loc[df1.id == 2, "dpred"].values[0]) == 16
    assert float(df1.loc[df1.id == 8, "dpred"].values[0]) == -4.0
    assert float(df1.loc[df1.id == 35, "dpred"].values[0]) == -10.0

    pyt.get_senescens_predictions("VDVI", 0.001, True)
    df1 = pyt.ldata

    assert int(df1.loc[df1.id == 1, "dpred"].values[0]) == -260
    assert int(df1.loc[df1.id == 2, "dpred"].values[0]) == 1283

    return


def test_save_fun():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )
    pyt.save("add_on/zarr_data/RGB_group.zip")
    pyt1 = read_zarr("add_on/zarr_data/RGB_group.zip")
    pyt1.save("add_on/zarr_data/RGB_group.zip")
    pyt2 = read_zarr("add_on/zarr_data/RGB_group.zip")

    assert pyt1.bands_name[0] == "red"
    assert pyt1.bands_name[1] == "green"
    assert pyt1.bands_name[2] == "blue"

    assert isinstance(pyt1.bands_name, list)
    assert isinstance(pyt1.raw_data, zarr.hierarchy.Group)
    assert isinstance(pyt1.ldata, pd.DataFrame)
    assert isinstance(pyt1.dates, list)
    assert isinstance(pyt1.n_bands, int)
    assert isinstance(pyt1.n_dates, int)

    assert pyt1.n_dates == 7
    assert pyt1.ldata.shape == (280, 7)
    assert pyt1.ldata.columns[0] == "id"
    assert pyt1.ldata.columns[1] == "fid"
    assert pyt1.ldata.columns[2] == "date"
    assert pyt1.ldata.columns[3] == "red"
    assert pyt1.ldata.columns[4] == "green"
    assert pyt1.ldata.columns[5] == "blue"

    assert pyt2.bands_name[0] == "red"
    assert pyt2.bands_name[1] == "green"
    assert pyt2.bands_name[2] == "blue"

    assert isinstance(pyt2.bands_name, list)
    assert isinstance(pyt2.raw_data, zarr.hierarchy.Group)
    assert isinstance(pyt2.ldata, pd.DataFrame)
    assert isinstance(pyt2.dates, list)
    assert isinstance(pyt2.n_bands, int)
    assert isinstance(pyt2.n_dates, int)

    assert pyt2.n_dates == 7
    assert pyt2.ldata.shape == (280, 7)
    assert pyt2.ldata.columns[0] == "id"
    assert pyt2.ldata.columns[1] == "fid"
    assert pyt2.ldata.columns[2] == "date"
    assert pyt2.ldata.columns[3] == "red"
    assert pyt2.ldata.columns[4] == "green"
    assert pyt2.ldata.columns[5] == "blue"

    return


def test_RGB_ind():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    pyt.RGB_VI(Red="red", Blue="blue", Green="green")

    assert pyt.ldata.shape[0] == 280
    assert pyt.ldata.shape[1] == 17

    assert "VDVI" in pyt.ldata.columns
    assert "NGRDI" in pyt.ldata.columns
    assert "VARI" in pyt.ldata.columns
    assert "GRRI" in pyt.ldata.columns
    assert "VEG" in pyt.ldata.columns
    assert "MGRVI" in pyt.ldata.columns
    assert "GLI" in pyt.ldata.columns
    assert "ExB" in pyt.ldata.columns
    assert "ExG" in pyt.ldata.columns
    assert "ExR" in pyt.ldata.columns

    assert (
        pyt.ldata.loc[pyt.ldata.id == 1, "VDVI"].values[0]
        == 0.1546743079777065
    )

    return
