# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel
# All rights reserved.

"""Provides the functions to read and extract the info from .tiff files."""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import rasterio
import cv2
from scipy import ndimage
import math
import os
import json
from shapely.geometry import MultiPolygon, Polygon
from rasterio import mask
from Pynomic.core import core
import re

# =============================================================================
# FUNCTIONS
# =============================================================================


def _read_grid(gpath):
    """Reads a geojson file.

    Args:
        gpath: grid path to a geojson file.

    Returns
    -------
        coordinate system of the grid.
        dict-like object with id and coords of each plot.
    """
    dics = {}
    if gpath.split(".")[1] == "geojson":
        plotgrids = open(gpath)
        plotgrids = json.load(plotgrids)
        crs_coords = plotgrids["crs"]["properties"]["name"]

        for p in plotgrids["features"]:
            dics[str(p["properties"]["fid"])] = p["geometry"]["coordinates"][0]

        return crs_coords, dics

    else:
        raise ValueError("Grid is not a geojson file")


def _get_tiff_files(fold_path):
    """Makes a list of tiff files of a folder path.

    Args:
        fold_path: folder path with the tiff files.

    Returns
    -------
        a list with the tiff files.
    """
    tiff_list = []
    for file in os.listdir(fold_path):

        if os.path.basename(file).split(".", 1)[1] == "tif":
            tiff_list.append(file)

    return tiff_list


def auto_fit_image(image, rot90=True, rimage=False, hbuffer=2, wbuffer=2):
    """Get's The parameters croop and angle automaticaly.

    Args:
        Image: an RGB Image or an array.
        rot90: boolean default True.
        rimage: rotation angle if wanted.
        hbuffer: number of pixels to buffer the image horizontaly.
        wbuffer: number of pixels to buffer the image horizontaly.

    Returns
    -------
        an image or parameters size of array to crop and angle to rotate.
    """
    # Gets the edges and generates a line to calculate the angle
    if isinstance(image, np.ndarray):
        gray = image
    else:
        gray = np.array(image.convert("L"))

    img_edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=10
    )

    # Calculates the angle
    angles = []
    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    angle = np.median(angles)

    # Applies the angle to the image and transforms it to horizonatal or
    # vertical
    if rot90 is True:
        rangle = 90 + angle

        if isinstance(image, np.ndarray):
            rot_im = ndimage.rotate(image.copy(), rangle)
        else:
            rot_im = image.copy().rotate(rangle, expand=True)

    else:
        rangle = angle
        if isinstance(image, np.ndarray):
            rot_im = ndimage.rotate(image.copy(), rangle)
        else:
            rot_im = image.copy().rotate(rangle, expand=True)

    # Gets the crooping parameters to eliminate the black background.
    if isinstance(image, np.ndarray):
        gry = rot_im
    else:
        gry = np.array(rot_im.convert("L"))

    blur = cv2.GaussianBlur(gry, (3, 3), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)

    if rimage:
        if isinstance(image, np.ndarray):
            return rot_im[
                (y + hbuffer) : (h - hbuffer), (x + wbuffer) : (w - wbuffer)
            ]
        else:
            return rot_im.crop(
                (x + wbuffer, y + hbuffer, w - wbuffer, h - hbuffer)
            )
    else:
        return (x + wbuffer, y + hbuffer, w - wbuffer, h - hbuffer), rangle


def _extract_bands_from_raster(raster_data, multiplot):
    """Separates the bands for the maks.

    Args:
        raster_data: a masked area of intrest(aoi).
        multiplot: a multiplot obj form sapely.

    Returns
    -------
        list with the bands array and an array from the mask.
    """
    masked_rast, aff = rasterio.mask.mask(
        raster_data, multiplot.geoms, crop=True
    )

    true_bands = masked_rast[:-1]
    masked_band = masked_rast[-1]

    return true_bands, masked_band


def extract_raster_data(raster_path, grid_path, bands_n=None):
    """Extracts the values from the raster file segregating each band and plot.

    Args:
        raster_path: path to raster file.
        grid_path: path to gird.
        bands_n:a list of the bands names and order.

    Returns
    -------
        dict with array of each band and plot.
        DataFrame with date, mean band for each plot.
        list bands name.
    """
    grid_crs, grids = _read_grid(grid_path)
    bands_mean = []
    array_dict = {}
    bands_name = []
    with rasterio.open(raster_path) as crs:
        for pos, g in enumerate(grids.keys()):
            if int(pos) == 0:
                coords = crs.meta["crs"]
                print(f"Raster Coords system: {coords}")
                print(f"Grid Coords system: {grid_crs}")

            # Diferentiate the true bands form the mask band.
            true_bands, masked_band = _extract_bands_from_raster(
                crs, MultiPolygon([Polygon(grids[g])])
            )

            # Get the fitting parameters.
            cpv, rangle = auto_fit_image(masked_band)

            # Enumerate the bands and name them.
            if bands_n:
                bands_name = bands_n
            else:
                n_band = np.array(range(0, len(true_bands))) + 1
                bands_name = ["band" + "_" + str(x) for x in n_band]

            # Fit the bands array with the parameters.
            fitted_bands = [
                ndimage.rotate(band, rangle)[cpv[1] : cpv[3], cpv[0] : cpv[2]]
                for band in true_bands
            ]

            # Save the mean for each band
            plot_fitted_bands = [
                np.mean(band.astype(float)) for band in fitted_bands
            ]
            mp_bands = []
            mp_bands.append(g)
            mp_bands.append(os.path.basename(raster_path).split("_")[0])
            for band in plot_fitted_bands:
                mp_bands.append(band)
            bands_mean.append(mp_bands)

            # Save the values in a dictionary.
            array_dict[g] = dict(zip(bands_name, fitted_bands))

    df = pd.DataFrame(bands_mean, columns=["id", "date", *bands_name])
    return array_dict, bands_name, df


def process_stack_tiff(folder_path, grid_path, bands_n=None):
    """Process all the .tiff files in a folder.

    Args:
        folder_path: folder that contains the .tiff files.
        grid_path: path of the geojson grid.
        bands_n: list like with the bands names ordered.

    Returns
    -------
        PynomicsProject object.
    """
    tif_list = _get_tiff_files(folder_path)
    features_dict = {}
    dates = []
    ldata = []
    for tiff_pos, tiff_file in enumerate(tif_list):
        print(f"{tiff_pos + 1}/{len(tif_list)} : {tiff_file}")
        if re.search(r"_", tiff_file):
            dates.append(tiff_file.split("_")[0])
        to_raw_data, bands_n, ldata_bands = extract_raster_data(
            folder_path + "/" + tiff_file, grid_path, bands_n
        )
        features_dict[tiff_file] = to_raw_data
        ldata.append(ldata_bands)
    df_data = pd.concat(ldata, axis=0)
    return core.Pynomicproject(
        raw_data=features_dict,
        ldata=df_data,
        dates=dates,
        n_dates=len(dates),
        bands_name=bands_n,
        n_bands=len(bands_n),
    )
