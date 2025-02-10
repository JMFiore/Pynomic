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

import shutil

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
    assert isinstance(pyt.raw_data, zarr.Group)
    print(type(pyt.raw_data))
    assert isinstance(pyt.ldata, pd.DataFrame)
    assert isinstance(pyt.dates, list)
    assert isinstance(pyt.n_bands, int)
    assert isinstance(pyt.n_dates, int)

    assert pyt.n_dates == 7
    assert pyt.ldata.shape == (280, 8)
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
    assert pyt.ldata.shape[1] == 8

    pyt.generate_unique_feature(VDVI_index, ["VDVI"], to_data=True)
    assert pyt.ldata.shape[1] == 9

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

    assert int(df1.loc[df1.id == 'A1', "dpred"].values[0]) == 13
    assert int(df1.loc[df1.id == 'A2', "dpred"].values[0]) == 16
    assert float(df1.loc[df1.id == 'A8', "dpred"].values[0]) == -4.0
    assert float(df1.loc[df1.id == 'A35', "dpred"].values[0]) == -10.0

    pyt.get_senescens_predictions("VDVI", 0.001, True)
    df1 = pyt.ldata

    assert int(df1.loc[df1.id == 'A1', "dpred"].values[0]) == -260
    assert int(df1.loc[df1.id == 'A2', "dpred"].values[0]) == 1283

    return


def test_save_fun():

    shutil.rmtree('add_on/zarr_data/RGB_group.zarr', ignore_errors=False)

    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )
    pyt.save("add_on/zarr_data/RGB_group.zarr")
    pyt1 = read_zarr("add_on/zarr_data/RGB_group.zarr")


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

    shutil.rmtree('add_on/zarr_data/RGB_group.zarr', ignore_errors=False)

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
    assert pyt.ldata.shape[1] == 18

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
        pyt.ldata.loc[pyt.ldata.id == 'A1', "VDVI"].values[0]
        == 0.1546743079777065
    )

    return


def test_Multispectral_VI():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/MFlights",
        "add_on/Grids/Grid_WheatData_UAV_2021_1.geojson",
        "ID",
        ["red", "green", "blue", "red_edge", "nir"],
    )

    assert pyt.ldata.shape[0] == 1100
    assert pyt.ldata.shape[1] == 21

    pyt.Multispectral_VI(
        Red="red", Blue="blue", Green="green", Red_edge="red_edge", Nir="nir"
    )

    assert pyt.ldata.shape[0] == 1100
    assert pyt.ldata.shape[1] == 38

    assert "NDVI" in pyt.ldata.columns
    assert "GNDVI" in pyt.ldata.columns
    assert "NDRE" in pyt.ldata.columns
    assert "EVI_2" in pyt.ldata.columns
    assert "SAVI" in pyt.ldata.columns
    assert "OSAVI" in pyt.ldata.columns
    assert "TDVI" in pyt.ldata.columns
    assert "NIRv" in pyt.ldata.columns
    assert "SR" in pyt.ldata.columns
    assert "SRredge" in pyt.ldata.columns
    assert "EVI" in pyt.ldata.columns
    assert "GNDRE" in pyt.ldata.columns
    assert "MCARI2" in pyt.ldata.columns
    assert "MTVI" in pyt.ldata.columns
    assert "MTVI2" in pyt.ldata.columns
    assert "RDVI" in pyt.ldata.columns
    assert "RTVI" in pyt.ldata.columns

    assert (
        pyt.ldata.loc[pyt.ldata.id == 'A1', "NDVI"].values[0]
        == 0.7110486695733146
    )

    return


def test_GLMC_TI():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/MFlights",
        "add_on/Grids/Grid_WheatData_UAV_2021_1.geojson",
        "ID",
        ["red", "green", "blue", "red_edge", "nir"],
    )

    assert pyt.ldata.shape[0] == 1100
    assert pyt.ldata.shape[1] == 21

    pyt.Calcualte_TI_GLCM([50], [90])

    assert pyt.ldata.shape[0] == 1100
    assert pyt.ldata.shape[1] == 46

    assert "red_50_90_cont" in pyt.ldata.columns
    assert "red_50_90_disst" in pyt.ldata.columns
    assert "red_50_90_homog" in pyt.ldata.columns
    assert "red_50_90_energy" in pyt.ldata.columns
    assert "red_50_90_corr" in pyt.ldata.columns
    assert "green_50_90_cont" in pyt.ldata.columns
    assert "green_50_90_disst" in pyt.ldata.columns
    assert "green_50_90_homog" in pyt.ldata.columns
    assert "green_50_90_energy" in pyt.ldata.columns
    assert "green_50_90_corr" in pyt.ldata.columns
    assert "blue_50_90_cont" in pyt.ldata.columns
    assert "blue_50_90_disst" in pyt.ldata.columns
    assert "blue_50_90_homog" in pyt.ldata.columns
    assert "blue_50_90_energy" in pyt.ldata.columns
    assert "blue_50_90_corr" in pyt.ldata.columns
    assert "red_edge_50_90_cont" in pyt.ldata.columns
    assert "red_edge_50_90_disst" in pyt.ldata.columns
    assert "red_edge_50_90_homog" in pyt.ldata.columns
    assert "red_edge_50_90_energy" in pyt.ldata.columns
    assert "red_edge_50_90_corr" in pyt.ldata.columns
    assert "nir_50_90_cont" in pyt.ldata.columns
    assert "nir_50_90_disst" in pyt.ldata.columns
    assert "nir_50_90_homog" in pyt.ldata.columns
    assert "nir_50_90_energy" in pyt.ldata.columns
    assert "nir_50_90_corr" in pyt.ldata.columns

    assert (
        round(pyt.ldata.loc[pyt.ldata.id == 'A1', "red_50_90_cont"].values[0], 0)
        == 7608.0
    )

    return
