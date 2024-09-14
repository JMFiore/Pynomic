# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024,  Fiore J.Manuel
# All rights reserved.

# =============================================================================
# IMPORTS
# =============================================================================
from Pynomic.core import core
from Pynomic.io import get_plot_bands

import numpy as np

import pandas as pd


import pytest

# =============================================================================
# FUNCTIONS
# =============================================================================


def test_Pnomicporject_obj():
    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        ["red", "green", "blue"],
    )

    assert isinstance(pyt, core.Pynomicproject)

    assert isinstance(pyt.bands_name, list)
    assert isinstance(pyt.raw_data, dict)
    assert isinstance(pyt.ldata, pd.DataFrame)
    assert isinstance(pyt.dates, list)
    assert isinstance(pyt.n_bands, int)
    assert isinstance(pyt.n_dates, int)

    assert pyt.n_dates == 7
    assert pyt.ldata.shape == (280, 5)
    assert pyt.ldata.columns[0] == "id"
    assert pyt.ldata.columns[1] == "date"
    assert pyt.ldata.columns[2] == "red"
    assert pyt.ldata.columns[3] == "green"
    assert pyt.ldata.columns[4] == "blue"
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
        ["red", "green", "blue"],
    )

    def VDVI_index(df):
        red = np.mean(df["red"])
        green = np.mean(df["green"])
        blue = np.mean(df["blue"])

        return (2 * green - red - blue) / (2 * green + red + blue)

    data = pyt.generate_unique_feature(VDVI_index, "VDVI")

    assert isinstance(data, pd.DataFrame)

    assert data.loc[0, "VDVI"] == 0.16568972306249757
    assert data.loc[3, "VDVI"] == 0.1308294903034819
    assert data.loc[275, "VDVI"] == 0.09725847938267855
    assert pyt.ldata.shape[1] == 5

    pyt.generate_unique_feature(VDVI_index, "VDVI", to_data=True)
    assert pyt.ldata.shape[1] == 6

    return
