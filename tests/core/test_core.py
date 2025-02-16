# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

# =============================================================================
# IMPORTS
# =============================================================================
import os
import shutil

import Pynomic
from Pynomic.core import core
from Pynomic.io import get_plot_bands


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
    assert pyt.dates[0] == "20180917"
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

    assert (
        data.loc[(data.id == "A1") & (data.date == "20180815"), "VDVI"].values[
            0
        ]
        == 0.1546743079777065
    )
    assert (
        data.loc[(data.id == "A3") & (data.date == "20180815"), "VDVI"].values[
            0
        ]
        == 0.15057698741228298
    )
    assert (
        data.loc[
            (data.id == "A23") & (data.date == "20180815"), "VDVI"
        ].values[0]
        == 0.11958853280011712
    )
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

    df1 = pyt.get_threshold_estimation("VDVI", 0.1)

    assert int(df1.loc[df1.id == "A1", "dpred"].values[0]) == 26
    assert int(df1.loc[df1.id == "A2", "dpred"].values[0]) == 27
    assert float(df1.loc[df1.id == "A8", "dpred"].values[0]) == -1
    assert float(df1.loc[df1.id == "A35", "dpred"].values[0]) == -2

    pyt.get_threshold_estimation("VDVI", 0.001, True)
    df1 = pyt.ldata

    assert int(df1.loc[df1.id == "A1", "dpred"].values[0]) == 39
    assert int(df1.loc[df1.id == "A2", "dpred"].values[0]) == 40

    return


def test_save_fun():

    dirlist = os.listdir("add_on/zarr_data")
    if "RGB_group" in dirlist:
        shutil.rmtree("add_on/zarr_data/RGB_group", ignore_errors=False)

    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )
    pyt.save("add_on/zarr_data/RGB_group")
    pyt1 = Pynomic.read_zarr("add_on/zarr_data/RGB_group")

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
    assert pyt1.ldata.shape == (280, 8)
    assert pyt1.ldata.columns[0] == "id"
    assert pyt1.ldata.columns[1] == "fid"
    assert pyt1.ldata.columns[2] == "date"
    assert pyt1.ldata.columns[3] == "red"
    assert pyt1.ldata.columns[4] == "green"
    assert pyt1.ldata.columns[5] == "blue"

    shutil.rmtree("add_on/zarr_data/RGB_group", ignore_errors=False)

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
        pyt.ldata.loc[
            (pyt.ldata.id == "A1") & (pyt.ldata.date == "20180917"), "VDVI"
        ].values[0]
        == 0.015838010015441387
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
        pyt.ldata.loc[
            (pyt.ldata.id == "A1") & (pyt.ldata.date == "20210628"), "NDVI"
        ].values[0]
        == 0.7110486695733146
    )
    assert np.round(
        pyt.ldata.loc[
            (pyt.ldata.id == "A1") & (pyt.ldata.date == "20210628"), "RTVI"
        ].values[0],
        0,
    ) == np.round(12.214456, 0)
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
        round(
            pyt.ldata.loc[
                (pyt.ldata.id == "A1") & (pyt.ldata.date == "20210628"),
                "red_50_90_cont",
            ].values[0],
            0,
        )
        == 7608.0
    )

    return


def test_calc_green_px():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    dat = pyt.Calcualte_green_pixels(
        Red="red", Blue="blue", Green="green", image_shape=(0, 180, 0, 45)
    )

    vals = dat.N_green_px + dat.N_non_green_px
    assert vals.values[0] == 8100
    assert (
        dat.loc[
            (dat.id == "A1") & (dat.date == "20180815"), "perc_green"
        ].values[0]
        == 0.77
    )

    return


def test_senescence_prediction_splines():
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

    df1 = pyt.get_senescens_Splines_predictions("VDVI", 0.1)

    assert int(df1.loc[df1.id == "A1", "dpred"].values[0]) == 24
    assert int(df1.loc[df1.id == "A2", "dpred"].values[0]) == 26
    assert float(df1.loc[df1.id == "A8", "dpred"].values[0]) == -1
    assert float(df1.loc[df1.id == "A35", "dpred"].values[0]) == -2

    pyt.get_senescens_Splines_predictions("VDVI", 0.001, True)
    df1 = pyt.ldata

    assert int(df1.loc[df1.id == "A1", "dpred"].values[0]) == 39
    assert int(df1.loc[df1.id == "A2", "dpred"].values[0]) == 40

    return


def test_senescence_prediction_loess():
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

    df1 = pyt.get_senescens_Loess_predictions("VDVI", 0.1)

    assert int(df1.loc[df1.id == "A1", "dpred"].values[0]) == 17
    assert int(df1.loc[df1.id == "A2", "dpred"].values[0]) == 20
    assert float(df1.loc[df1.id == "A8", "dpred"].values[0]) == -1
    assert float(df1.loc[df1.id == "A35", "dpred"].values[0]) == -3

    pyt.get_senescens_Loess_predictions("VDVI", 0.001, to_data=True)
    df1 = pyt.ldata

    assert int(df1.loc[df1.id == "A1", "dpred"].values[0]) == 43
    assert int(df1.loc[df1.id == "A2", "dpred"].values[0]) == 44

    return


def test_save_plots_fun():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_small.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    dirlist = os.listdir("add_on/zarr_data")
    if "imgs" in dirlist:
        shutil.rmtree("add_on/zarr_data/imgs", ignore_errors=False)

    if "tiffs" in dirlist:
        shutil.rmtree("add_on/zarr_data/tiffs", ignore_errors=False)

    os.mkdir("add_on/zarr_data/imgs")
    os.mkdir("add_on/zarr_data/tiffs")

    def rgb_fun(df):

        red = df["red"][:]
        green = df["green"][:]
        blue = df["blue"][:]

        return np.dstack([red, green, blue])

    pyt.save_indiv_plots_images(
        "add_on/zarr_data/imgs",
        fun=rgb_fun,
        identification_col="id",
        file_type="jpg",
    )

    imgslist = os.listdir("add_on/zarr_data/imgs")

    assert len(imgslist) == len(pyt.dates)
    for f in imgslist:
        pth = "add_on/zarr_data/imgs" + "/" + f
        assert len(os.listdir(pth)) == 4
        assert "A1.jpg" in os.listdir(pth)
        assert "A3.jpg" in os.listdir(pth)

    shutil.rmtree("add_on/zarr_data/imgs", ignore_errors=False)

    pyt.save_indiv_plots_images(
        "add_on/zarr_data/tiffs",
        fun=rgb_fun,
        identification_col="id",
        file_type="tiff",
    )

    assert len(imgslist) == len(pyt.dates)
    for f in imgslist:
        pth = "add_on/zarr_data/tiffs" + "/" + f
        assert len(os.listdir(pth)) == 4
        assert "A1.tiff" in os.listdir(pth)
        assert "A3.tiff" in os.listdir(pth)

    shutil.rmtree("add_on/zarr_data/tiffs", ignore_errors=False)

    return
