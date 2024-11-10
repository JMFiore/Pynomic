# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

"""Provides the functions to read and extract the info from .tiff files."""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import rasterio
import cv2
import os
import json
from shapely.geometry import MultiPolygon, Polygon
from rasterio import mask
from Pynomic.core import core
import re
from PIL import Image
import zarr
import io
import pandas_geojson as pdg

# =============================================================================
# FUNCTIONS
# =============================================================================


def _read_grid(gpath, col_id: str):
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
            dics[str(p["properties"][col_id])] = p["geometry"]["coordinates"][
                0
            ]

        return crs_coords, dics

    else:
        raise ValueError("Grid is not a geojson file")


def _get_dataframe_from_json(path_gjson):
    data = pdg.read_geojson(path_gjson)
    collist = data.get_properties()
    dfg = data.to_dataframe()
    keep = []
    for c in dfg.columns:
        for m in collist:
            if len(c.split("."+m)) > 1:
                keep.append(c)
    dfa = dfg.loc[:, keep].copy()
    dfa.columns = collist
    return dfa


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


def auto_fit_image(mtx, hbuffer=2, wbuffer=2):
    """Takes an array and returns the crooping and angle parameters.

    Args
        mtx: np.array
        hbuffer: int. height buffer.
        wbuffer: int. with buffer.

    Returns
    -------
        tuple with croping parameters. Angle value.
    """

    gray = np.where(mtx <= 0, mtx, 255)

    gray = np.uint8(np.where(gray > 0, gray, 0))
    gray = np.array(gray)

    # edges = cv2.Canny(gray,150,150*2)

    contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    rect = cv2.minAreaRect(contours[-1])

    if rect[1][0] < rect[1][1]:
        angle = rect[2]
        gry = np.array(Image.fromarray(gray).rotate(angle, expand=True))
    else:
        angle = rect[2] + 90
        gry = np.array(Image.fromarray(gray).rotate(angle, expand=True))

    blur = cv2.GaussianBlur(gry, (3, 3), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)

    h1 = y + hbuffer
    h2 = y + (h - hbuffer)
    w1 = x + wbuffer
    w2 = x + (w - wbuffer)
    return (h1, h2, w1, w2), angle


def _extract_bands_from_raster(raster_data, multiplot, alpha_idx=-1):
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

    if alpha_idx == -1:
        true_bands = masked_rast.copy()
        masked_band = masked_rast[-1]
    else:
        true_bands = list(masked_rast)
        masked_band = true_bands.pop(alpha_idx)

    return true_bands, masked_band


def extract_raster_data(raster_path, grid_path, col_id: str, bands_n=None):
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
    grid_cs, grids = _read_grid(grid_path, col_id)
    bands_mean = []
    array_dict = {}
    bands_name = []
    with rasterio.open(raster_path) as src:
        for pos, g in enumerate(grids.keys()):
            if int(pos) == 0:
                coords = src.meta["crs"]
                print(f"Raster Coords system: {coords}")
                print(f"Grid Coords system: {grid_cs}")

            # Check if contains alpha band
            contains_alpha_band = -1
            for idx, interp in enumerate(src.colorinterp, start=1):
                if interp == rasterio.enums.ColorInterp.alpha:
                    contains_alpha_band = idx - 1

            if contains_alpha_band != -1:
                # Diferentiate the true bands form the mask band.
                true_bands, masked_band = _extract_bands_from_raster(
                    src, MultiPolygon([Polygon(grids[g])]), contains_alpha_band
                )
            else:
                # Returns the las band for fiting.
                true_bands, masked_band = _extract_bands_from_raster(
                    src, MultiPolygon([Polygon(grids[g])])
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
                np.array(Image.fromarray(band).rotate(rangle, expand=True))[
                    cpv[0] : cpv[1], cpv[2] : cpv[3]
                ]
                for band in true_bands
            ]

            # Save the mean for each band
            plot_fitted_bands = [
                np.mean(band.astype(float)) for band in fitted_bands
            ]
            mp_bands = []
            # Numerical id for project.
            mp_bands.append(pos + 1)
            # Original id from the grid can be numerical or text or both.
            mp_bands.append(g)
            # gets the date
            mp_bands.append(os.path.basename(raster_path).split("_")[0])
            for band in plot_fitted_bands:
                mp_bands.append(band)
            bands_mean.append(mp_bands)

            # Save the values in a dictionary.
            array_dict[pos + 1] = dict(zip(bands_name, fitted_bands))

    df = pd.DataFrame(bands_mean, columns=["id", col_id, "date", *bands_name])
    dat = _get_dataframe_from_json(path_gjson=grid_path)
    dat[col_id] = dat[col_id].astype(str)

    df = df.merge(dat, on=col_id)
    return array_dict, bands_name, df


def process_stack_tiff(folder_path, grid_path, col_id: str, bands_n=None):
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
    raw_data = zarr.group()
    raw_data.create_group("dates")
    dates = []
    ldata = []
    date_key = ""
    for tiff_pos, tiff_file in enumerate(tif_list):
        print(f"{tiff_pos + 1}/{len(tif_list)} : {tiff_file}")
        if re.search(r"_", tiff_file):
            date_key = tiff_file.split("_")[0]
            dates.append(date_key)

        to_raw_data, bands_n, ldata_bands = extract_raster_data(
            folder_path + "/" + tiff_file, grid_path, col_id, bands_n
        )

        raw_data["dates"].create_group(date_key)

        for plot_id in to_raw_data.keys():
            raw_data["dates"][date_key].create_group(plot_id)
            for band in to_raw_data[plot_id].keys():
                raw_data["dates"][date_key][plot_id].create_group(band)
                raw_data["dates"][date_key][plot_id][band] = to_raw_data[
                    plot_id
                ][band]

        ldata.append(ldata_bands)

    df_data = pd.concat(ldata, axis=0)

    return core.Pynomicproject(
        raw_data=raw_data,
        ldata=df_data,
        dates=dates,
        n_dates=len(dates),
        bands_name=bands_n,
        n_bands=len(bands_n),
    )


def read_zarr(path):
    """Reads a zarr project previously saved from pynomics.

    Args
        path: a path to the zarr folder

    Returns
    -------
        Pynomicproject object
    """
    store = zarr.open(path, mode="a")
    info = zarr.group()
    zarr.copy_all(store, info)
    df_buffer = io.BytesIO()
    df_buffer.write(info["ldata"][0])
    ldata = pd.read_parquet(df_buffer)

    return core.Pynomicproject(
        raw_data=info,
        ldata=ldata.copy(),
        n_dates=len(ldata.date.unique()),
        dates=list(ldata.date.unique()),
        n_bands=len(info.bands_name[:]),
        bands_name=list(info.bands_name[:].copy()),
    )
