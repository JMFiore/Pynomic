# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

"""Provides the objects and functions."""

# =============================================================================
# IMPORTS
# =============================================================================
from Pynomic.io import get_plot_bands


import numpy as np

# =============================================================================
# CLASSES
# =============================================================================


def test_image_timeline():

    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    def VDVI_inex(df):
        red = np.mean(df["red"])
        green = np.mean(df["green"])
        blue = np.mean(df["blue"])

        return [(2 * green - red - blue) / (2 * green + red + blue)]

    def VDVI_inex2(df):
        red = np.float16(df["red"])
        green = np.float16(df["green"])
        blue = np.float16(df["blue"])

        return (2 * green - red - blue) / (2 * green + red + blue)

    pyt.generate_unique_feature(VDVI_inex, ["VDVI"], to_data=True)

    at, atb = pyt.plot.image_timeline("VDVI", "A3", VDVI_inex2)

    assert atb.get_xlabel() == "Flight dates"
    assert atb.get_title() == ""
    assert atb.get_ylabel() == "VDVI"
    assert atb.get_yscale() == "linear"

    atd, atbd = pyt.plot.image_timeline("VDVI", "A3", VDVI_inex2, days=True)
    assert atbd.get_xlabel() == "Flight days"
    return


def test_timeline():

    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    def VDVI_inex(df):
        red = np.mean(df["red"])
        green = np.mean(df["green"])
        blue = np.mean(df["blue"])

        return [(2 * green - red - blue) / (2 * green + red + blue)]

    pyt.generate_unique_feature(VDVI_inex, ["VDVI"], to_data=True)

    axis = pyt.plot.timeline("VDVI", "A3")

    assert axis.get_xlabel() == "Flight dates"
    assert axis.get_title() == "VDVI - Plot Id A3"
    assert axis.get_ylabel() == "VDVI"
    assert axis.get_yscale() == "linear"

    axis_days = pyt.plot.timeline("VDVI", "A3", days=True)
    assert axis_days.get_xlabel() == "Flight days"
    return


def test_timeline_RGB():

    pyt = get_plot_bands.process_stack_tiff(
        "add_on/flights",
        "add_on/Grids/Labmert_test_grid.geojson",
        "fid",
        ["red", "green", "blue"],
    )

    def VDVI_inex(df):
        red = np.mean(df["red"])
        green = np.mean(df["green"])
        blue = np.mean(df["blue"])

        return [(2 * green - red - blue) / (2 * green + red + blue)]

    pyt.generate_unique_feature(VDVI_inex, ["VDVI"], to_data=True)

    axis = pyt.plot.RGB_image_timeline(
        "VDVI", "A3", Red="red", Green="green", Blue="blue", days=True
    )

    assert axis[1].get_xlabel() == "Flight days"
    assert axis[1].get_title() == ""
    assert axis[1].get_ylabel() == "VDVI"
    assert axis[1].get_yscale() == "linear"

    axis = pyt.plot.RGB_image_timeline(
        "VDVI",
        "A3",
        Red="red",
        Green="green",
        Blue="blue",
        days=True,
        Size=(0, 180, 0, 45),
    )

    assert axis[1].get_xlabel() == "Flight days"
    assert axis[1].get_title() == ""
    assert axis[1].get_ylabel() == "VDVI"
    assert axis[1].get_yscale() == "linear"

    return
