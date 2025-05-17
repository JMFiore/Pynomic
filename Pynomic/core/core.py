# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

"""Provides the objects and functions."""

# =============================================================================
# IMPORTS
# =============================================================================
import attrs


import pandas as pd
import numpy as np
import zarr
from sklearn.linear_model import LinearRegression
import os
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import json

# =============================================================================
# CLASSES
# =============================================================================


@attrs.define
class Pynomicproject:
    """Contains all the extracted bands from each plot and dates.

    Parameters
    ----------
    raw_data :zarr.hierarchy.Group
        contains all the data.

    ldata :Pandas Dataframe
        contains all the procesed data.
    """

    raw_data: zarr.Group
    ldata: pd.DataFrame
    n_dates: int
    dates: list
    n_bands: int
    bands_name: list

    def __getitem__(self, k: str):
        """Allow attribute access using dictionary-like syntax.

        Parameters
        ----------
        k :str
            Attribute name.

        Returns
        -------
        Any
            Value of the attribute.

        Raises
        ------
        KeyError
            If the attribute does not exist.
        """
        try:
            return getattr(self, k)
        except AttributeError:
            raise KeyError(k)

    def RGB_VI(self, Red, Blue, Green):
        """Calculates RGB Vegetation index.

        Parameters
        ----------
        Red:str
            name of the column that contains the red band.
        Blue:str
            name of the column that contains the blue band.
        Green:str
            name of the column that contains the green band.

        Returns
        -------
            geodataframe
        """
        df = self.ldata
        red = df.loc[:, Red]
        blue = df.loc[:, Blue]
        green = df.loc[:, Green]

        # Visible-band difference vegetation index
        df["VDVI"] = (2 * green - red - blue) / (2 * green + red + blue)
        # Normalized green–red difference index (Kawashima Index) also called GRVI
        df["NGRDI"] = (green - red) / (green + red)
        # Visible Atmospherically Resistant Index
        df["VARI"] = (green - red) / (green + red - blue)
        # Green–red ratio index
        df["GRRI"] = green / red
        # Vegetativen
        df["VEG"] = green / ((red**0.667) * (blue ** (1 - 0.667)))
        # Modified Green Red Vegetation Index
        df["MGRVI"] = ((green**2) - (red**2)) / ((green**2) + (blue**2))
        # Green Leaf Index
        df["GLI"] = (2 * green - red - blue) / ((-red) - blue)
        # Excess Red Vegetation Index
        df["ExR"] = (1.4 * red - green) / (green + red + blue)
        # Excess Blue Vegetation Index
        df["ExB"] = (1.4 * blue - green) / (green + red + blue)
        # Excess Green Vegetation Index
        df["ExG"] = 2 * green - red - blue

        return

    def Multispectral_VI(self, Red, Blue, Green, Red_edge, Nir):
        """Calculates Multispectral Vegetation index.

        Parameters
        ----------
        Red:str
            name of the column that contains the red band.
        Blue:str
            name of the column that contains the blue band.
        Green:str
            name of the column that contains the green band.
        Red_edge:str
            name of the column that contains the Red edge band.
        NIR:str
            name of the column that contains thee NIR band.

        Returns
        -------
            geodataframe
        """
        df = self.ldata
        red = df.loc[:, Red]
        blue = df.loc[:, Blue]
        green = df.loc[:, Green]
        redge = df.loc[:, Red_edge]
        nir = df.loc[:, Nir]

        # NDVI index
        df["NDVI"] = (nir - red) / (nir + red)
        # GNDVI
        df["GNDVI"] = (nir - green) / (nir + green)
        # NDRE
        df["NDRE"] = (nir - redge) / (nir + redge)
        # EVI_2
        df["EVI_2"] = 2.5 * ((nir - red) / (nir + (2.4 * red) + 1))
        # SAVI
        df["SAVI"] = (1.5 * (nir - red)) / (nir + red + 0.5)
        # OSAVI
        df["OSAVI"] = (1.16 * (nir - red)) / (nir + red + 0.16)
        # TDVI
        df["TDVI"] = np.sqrt((0.5 + ((nir - red) / (nir + red))))
        # NIRV
        df["NIRv"] = nir * ((nir - red) / (nir + red))
        # Simple ratio
        df["SR"] = nir / red
        # SRredge Simple ratio Red edge
        df["SRredge"] = nir / redge
        # EVI Enhanced vegetation index
        df["EVI"] = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        # GNDRE Green Normalized difference Red Edge index
        df["GNDRE"] = (redge - green) / (redge + green)
        # MCARI2 Modified Chlorophyll Absorption Ratio Index
        df["MCARI2"] = 1.5 * (
            (2.5 * (nir - red))
            - (1.3 * (nir - green))
            / np.sqrt(((((2 * nir) + 1) ** 2) - ((6 * nir) - (5 * red))) - 0.5)
        )
        # MTVI Modified Triangular Vegetation Index
        df["MTVI"] = 1.2 * ((1.2 * (nir - green)) - (2.5 * (red - green)))
        # MTVI2  Modified Triangular Vegetation Index
        df["MTVI2"] = (
            1.5 * ((1.2 * (nir - green)) - (2.5 * (red - green)))
        ) / np.sqrt(((((2 * nir) + 1) ** 2) - ((6 * nir) - (5 * red))) - 0.5)
        # NDRE Normalized Difference Red Edge index
        df["NDRE"] = (nir - redge) / (nir + redge)
        # RDVI Renormalized Difference Vegetation Index
        df["RDVI"] = (nir - red) / np.sqrt(nir - red)
        # RTVI Red Edge Triangulated vegetation Index
        df["RTVI"] = (100 * (nir - redge)) - (10 * (nir - green))

        return

    def Calcualte_TI_GLCM(self, distances: list, angles: list):
        """Calculates texturial indices from bands.

        be aweare the O = (n_dist * n_bands)^n_angles.
        time and number of variables can scale very quckly.

        Parameters
        ----------
        distances:list
            list of distances to work usaly 2 or 3 .
        algles:lsit
            list of angles to work.

        Returns
        -------
            geodataframe.
        """

        def _calculate_GLCM(df, angles, distances, bands):
            features_names = []
            glcm_values = []
            for angl in angles:
                for dist in distances:
                    for b in bands:
                        gray = df[b][:].copy()
                        if not np.issubdtype(gray.dtype, np.uint8):
                            gray *= 255 / np.round(gray, 6).max()
                            gray = np.uint8(np.round(gray, 0).astype(int))
                        glcm = graycomatrix(
                            gray,
                            distances=[dist],
                            angles=[angl],
                            levels=256,
                            symmetric=True,
                            normed=True,
                        )
                        features_names.append(
                            b
                            + "_"
                            + str(dist)
                            + "_"
                            + str(angl)
                            + "_"
                            + "cont"
                        )
                        glcm_values.append(
                            round(graycoprops(glcm, "contrast")[0][0], 4)
                        )
                        features_names.append(
                            b
                            + "_"
                            + str(dist)
                            + "_"
                            + str(angl)
                            + "_"
                            + "disst"
                        )
                        glcm_values.append(
                            round(graycoprops(glcm, "dissimilarity")[0][0], 4)
                        )
                        features_names.append(
                            b
                            + "_"
                            + str(dist)
                            + "_"
                            + str(angl)
                            + "_"
                            + "homog"
                        )
                        glcm_values.append(
                            round(graycoprops(glcm, "homogeneity")[0][0], 4)
                        )
                        features_names.append(
                            b
                            + "_"
                            + str(dist)
                            + "_"
                            + str(angl)
                            + "_"
                            + "energy"
                        )
                        glcm_values.append(
                            round(graycoprops(glcm, "energy")[0][0], 4)
                        )
                        features_names.append(
                            b
                            + "_"
                            + str(dist)
                            + "_"
                            + str(angl)
                            + "_"
                            + "corr"
                        )
                        glcm_values.append(
                            round(graycoprops(glcm, "correlation")[0][0], 4)
                        )
            return glcm_values, features_names

        values_list = []
        for flight_date in self.dates:
            for plot in self.raw_data["dates"][flight_date].group_keys():
                bands_names = []
                bands_arr = []
                for band in self.bands_name:
                    bands_names.append(band)
                    bands_arr.append(
                        self.raw_data["dates"][flight_date][plot][band][:]
                    )
                values, features_names = _calculate_GLCM(
                    df=dict(zip(bands_names, bands_arr)),
                    distances=distances,
                    angles=angles,
                    bands=self.bands_name,
                )
                values.insert(0, plot)
                values.insert(1, flight_date)
                values_list.append(values)

        features_names.insert(0, "id")
        features_names.insert(1, "date")

        tidf = pd.DataFrame(values_list, columns=features_names)
        # tidf.id = tidf.id.astype(int)
        self.ldata = self.ldata.merge(tidf, on=["id", "date"])
        return self.ldata

    def Calcualte_green_pixels(
        self,
        Red: str,
        Blue: str,
        Green: str,
        image_shape: tuple,
        min_val=30,
        max_val=75,
        to_data=False,
    ):
        """Extracts the green and non-green pixels from each image HSL.

        Parameters
        ----------
        Red:str
            name of the column that contains the red band.
        Blue:str
            name of the column that contains the blue band.
        Green:str
            name of the column that contains the green band.
        image_shape:tuple
            (top, bottom, left, right) indicates the area
        min_val:int
            in HUE range.
        max_val:int
            in HUE range

        Returns
        -------
            geodataframe.
        """

        def _calculate_grpx(
            dicmtx,
            red: str,
            green: str,
            blue: str,
            min_val: int,
            max_val: int,
            im_shp=image_shape,
        ):

            if len(im_shp) < 4:
                red1 = dicmtx[red][:]
                red1 = cv2.normalize(
                    red1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
                )
                green1 = dicmtx[green][:]
                green1 = cv2.normalize(
                    green1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
                )
                blue1 = dicmtx[blue][:]
                blue1 = cv2.normalize(
                    blue1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
                )
            else:
                red1 = dicmtx[red][
                    im_shp[0] : im_shp[1], im_shp[2] : im_shp[3]
                ]
                red1 = cv2.normalize(
                    red1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
                )
                green1 = dicmtx[green][
                    im_shp[0] : im_shp[1], im_shp[2] : im_shp[3]
                ]
                green1 = cv2.normalize(
                    green1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
                )
                blue1 = dicmtx[blue][
                    im_shp[0] : im_shp[1], im_shp[2] : im_shp[3]
                ]
                blue1 = cv2.normalize(
                    blue1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
                )

            img = np.dstack([red1, green1, blue1])
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (min_val, 25, 25), (max_val, 255, 255))

            unique, counts = np.unique(mask, return_counts=True)
            unique = unique.astype(int).tolist()
            counts = counts.astype(int).tolist()
            ab = dict(zip(unique, counts))
            if len(unique) > 1:
                val255 = mask == 255
                val255 = int(val255.sum())
                val0 = mask == 0
                val0 = int(val0.sum())
                por = val255 / (mask.shape[0] * mask.shape[1])
                return [np.round(por, 2), val255, val0]
            else:
                if 0 in unique:
                    nongr = ab.get(0)
                    return [0, 0, nongr]
                if 255 in unique:
                    gp = ab.get(255)
                    return [1, gp, 0]

        values_list = []
        for flight_date in self.dates:
            for plot in self.raw_data["dates"][flight_date].group_keys():
                bands_names = []
                bands_arr = []
                for band in self.bands_name:
                    bands_names.append(band)
                    bands_arr.append(
                        self.raw_data["dates"][flight_date][plot][band][:]
                    )
                values = _calculate_grpx(
                    dicmtx=dict(zip(bands_names, bands_arr)),
                    red=Red,
                    green=Green,
                    blue=Blue,
                    min_val=min_val,
                    max_val=max_val,
                )
                values.insert(0, plot)
                values.insert(1, flight_date)
                values_list.append(values)

        features_names = [
            "id",
            "date",
            "perc_green",
            "N_green_px",
            "N_non_green_px",
        ]

        tidf = pd.DataFrame(values_list, columns=features_names)
        # tidf.id = tidf.id.astype(int)

        if to_data:
            self.ldata = self.ldata.merge(tidf, on=["id", "date"])
            return self.ldata
        else:
            return tidf

    def generate_unique_feature(
        self, function, features_names: list, to_data=False
    ):
        """Higher order function that iterate through the flight dates.

        Parameters
        ----------
        function :function
            function that contains a formula and
            returns a list.
        new_name :list
            name of the new features.
        to_data :bool
            merges it with the project data.

        Returns
        -------
            geodataframe.
        """
        if isinstance(features_names, list):
            values_list = []
            for flight_date in self.dates:
                for plot in self.raw_data["dates"][flight_date].group_keys():
                    bands_names = []
                    bands_arr = []
                    for band in self.bands_name:
                        bands_names.append(band)
                        bands_arr.append(
                            self.raw_data["dates"][flight_date][plot][band][:]
                        )
                    values = function(dict(zip(bands_names, bands_arr)))
                    values.insert(0, plot)
                    values.insert(1, flight_date)
                    values_list.append(values)

            features_names.insert(0, "id")
            features_names.insert(1, "date")

            if to_data:

                df = pd.DataFrame(values_list, columns=features_names)
                # df.id = df.id.astype(int)
                self.ldata = self.ldata.merge(df, on=["id", "date"])
                return self.ldata

            else:

                return pd.DataFrame(values_list, columns=features_names)
        else:
            return print("feature_names is not a list")

    def get_threshold_estimation(
        self, band: str, threshold: float, to_data: bool = False, from_day=0
    ):
        """Generates predictions of senecense by providing threshold and index.

        Parameters
        ----------
        band:str
            Band name to be used in the prediciton.
        threshold:float
            value to determen if a plot is dry or not.
        to_data:bool
            boolean value to save or not the predictions.

        Returns
        -------
            Geodataframe
        """

        def _case_in(plot, col_val, numerical_date_col, threshold):
            plot = plot.sort_values(
                numerical_date_col, ascending=True
            ).reset_index()
            for plotpos, plotval in enumerate(plot[numerical_date_col].values):
                if (
                    plot.loc[
                        plot[numerical_date_col] == plotval, col_val
                    ].values[0]
                    <= threshold
                ) & (plotpos != 0):

                    if (
                        plot.loc[
                            plot[numerical_date_col] == plotval, col_val
                        ].values[0]
                        == threshold
                    ):
                        return round(plotval)
                    else:
                        ant_date = plot[numerical_date_col].values[plotpos - 1]
                        colant_val = plot.loc[
                            plot[numerical_date_col] == ant_date, col_val
                        ].values[0]
                        col_value = plot.loc[
                            plot[numerical_date_col] == plotval, col_val
                        ].values[0]
                        yval = np.array([ant_date, plotval]).reshape(-1, 1)
                        xval = np.array([colant_val, col_value]).reshape(-1, 1)
                        lm = LinearRegression().fit(xval, yval)
                        plotpred = lm.predict(
                            np.array([threshold]).reshape(-1, 1)
                        )[0][0]

                        return round(plotpred)
            return -999

        def _case_upper(plot, col_val, numerical_date_col, threshold):
            plot = plot.sort_values(
                numerical_date_col, ascending=True
            ).reset_index()
            x = plot[numerical_date_col].values
            y = plot[col_val].values

            spl = UnivariateSpline(x, y, k=3, s=4)
            # xs = np.linspace(x.min(), x.max(), 1000)

            def _func(x_val):
                return spl(x_val) - threshold

            initial_guess = x.astype(int).min()
            result = root(_func, initial_guess)

            if result.success:
                plotpred = result.x[0]
            else:
                plotpred = 0

            return round(plotpred)

        def _case_lower(plot, col_val, numerical_date_col, threshold):
            plot = plot.sort_values(
                numerical_date_col, ascending=True
            ).reset_index()
            x = plot[numerical_date_col].values
            y = plot[col_val].values

            spl = UnivariateSpline(x, y, k=3, s=4)
            # xs = np.linspace(x.min(), x.max(), 1000)

            def _func(x_val):
                return spl(x_val) - threshold

            initial_guess = x.astype(int).max()
            result = root(_func, initial_guess)

            if result.success:
                plotpred = result.x[0]
            else:
                plotpred = -997

            return round(plotpred)

        df1 = self.ldata.copy()
        plot_id_col = "id"
        col_val = band
        df1["num_day"] = (
            pd.to_datetime(df1.date) - pd.to_datetime(df1.date).min()
        )
        df1["num_day"] = (
            df1["num_day"].astype(str).apply(lambda x: int(x.split(" ")[0]))
        )
        numerical_date_col = "num_day"

        if from_day > 0:
            df1 = df1.loc[df1.num_day > from_day].copy()

        for p in df1[plot_id_col].unique():

            plot = df1.loc[df1[plot_id_col] == p]

            # First case if threshold is in rage
            if (plot[col_val].min() <= threshold) & (
                plot[col_val].values[: int((len(plot[col_val]) / 2))].max()
                >= threshold
            ):
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_in(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "IN"

            # Second case if threshold is upper than the range in col_val
            elif (
                plot[col_val].values[: int((len(plot[col_val]) / 2))].max()
                < threshold
            ):
                print(f"Plot Id: {p} threshold is upper than range")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_upper(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "upper"

            # Third case if threshold is lower than the range in col_val
            elif plot[col_val].min() >= threshold:
                print(f"Plot Id: {p} threshold is lower than range")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_lower(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "lower"

        if to_data:
            self.ldata = self.ldata.merge(
                df1.loc[
                    :,
                    [
                        "id",
                        numerical_date_col,
                        "dpred",
                        "in_range",
                    ],
                ],
                on=["id"],
                how="left",
            )
        else:
            return df1

    @property
    def plot(self):
        """Generate plots from spectra."""
        from .plot import Pynomicplotter

        return Pynomicplotter(self)

    def save(self, path):
        """Function to save project in a directory.

        Parameters
        ----------
        path:str
            Name of the directory.

        Returns
        -------
            A directory with the Pynomicproject folders.
        """
        out_store = zarr.DirectoryStore(path + "/" + "raw_data")
        zarr.copy_store(self.raw_data.store, out_store)

        self.ldata.to_file(path + "/" + "ldata.shp", driver="ESRI Shapefile")
        prop_dic = {
            "dates": self.dates,
            "bands": self.bands_name,
        }
        with open(path + "/" + "obj_properties.json", mode="w") as outfile:
            json.dump(prop_dic, outfile)

        return

    def save_indiv_plots_images(
        self, folder_path, fun, identification_col, file_type: str
    ):
        """Creates as many folders as dates in path provided and saves the plot images.

        Parameters
        ----------
        folder_path:str
            Path where to save the images.
        fun:function
            function to use to stack the bands.
        identification_col:str
            Column of ldata where the ids are.
        file_type:str
            tiff or jpg
        Returns
        -------
            folder with images.
        """
        for d in self.dates:
            path = os.path.join(folder_path, d)
            os.mkdir(path)
            for p in self.raw_data["dates"][d].group_keys():
                bands_names = []
                bands_arr = []
                for band in self.bands_name:
                    bands_names.append(band)
                    bands_arr.append(self.raw_data["dates"][d][p][band][:])
                arrays = fun(dict(zip(bands_names, bands_arr)))
                name = str(
                    self.ldata.loc[
                        self.ldata["id"] == p, identification_col
                    ].unique()[0]
                )
                if file_type == "tiff":
                    image_path = os.path.join(path, name + ".tiff")
                if file_type == "jpg":
                    image_path = os.path.join(path, name + ".jpg")
                image = Image.fromarray(arrays)
                image.save(image_path)

    def get_senescens_Splines_predictions(
        self, band: str, threshold: float, to_data: bool = False, from_day=0
    ):
        """Generates predictions of senecense by providing threshold using the spline method.

        Parameters
        ----------
        band:str
            Band name to be used in the prediciton.
        threshold:float
            value to determen if a plot is dry or not.
        to_data:bool
            boolean value to save or not the predictions.

        Returns
        -------
            Geodataframe
        """

        def _case_in(plot, col_val, numerical_date_col, threshold):
            plot = plot.sort_values(numerical_date_col, ascending=True)
            x = plot[numerical_date_col].values
            y = plot[col_val].values

            spl = UnivariateSpline(x, y, k=3, s=4)
            # xs = np.linspace(x.min(), x.max(), 1000)

            def _func(x_val):
                return spl(x_val) - threshold

            # # INITIAL GUESS ESTIMATOR ##
            def _inestim(plot, col_val, numerical_date_col, threshold):

                for plotpos, plotval in enumerate(
                    plot[numerical_date_col].values
                ):
                    if (
                        plot.loc[
                            plot[numerical_date_col] == plotval, col_val
                        ].values[0]
                        <= threshold
                    ) & (plotpos != 0):

                        if (
                            plot.loc[
                                plot[numerical_date_col] == plotval, col_val
                            ].values[0]
                            == threshold
                        ):
                            return round(plotval)
                        else:
                            ant_date = plot[numerical_date_col].values[
                                plotpos - 1
                            ]
                            colant_val = plot.loc[
                                plot[numerical_date_col] == ant_date, col_val
                            ].values[0]
                            col_value = plot.loc[
                                plot[numerical_date_col] == plotval, col_val
                            ].values[0]
                            yval = np.array([ant_date, plotval]).reshape(-1, 1)
                            xval = np.array([colant_val, col_value]).reshape(
                                -1, 1
                            )
                            lm = LinearRegression().fit(xval, yval)
                            plotpred = lm.predict(
                                np.array([threshold]).reshape(-1, 1)
                            )[0][0]

                            return round(plotpred)
                return -900

            initial_guess = _inestim(
                plot=plot,
                col_val=col_val,
                numerical_date_col=numerical_date_col,
                threshold=threshold,
            )

            result = root(_func, initial_guess)

            if result.success:
                plotpred = result.x[0]
            else:
                plotpred = 0

            return round(plotpred)

        def _case_upper(plot, col_val, numerical_date_col, threshold):
            plot = plot.sort_values(
                numerical_date_col, ascending=True
            ).reset_index()
            x = plot[numerical_date_col].values
            y = plot[col_val].values

            spl = UnivariateSpline(x, y, k=3, s=4)
            # xs = np.linspace(x.min(), x.max(), 1000)

            def _func(x_val):
                return spl(x_val) - threshold

            initial_guess = x.astype(int).min()
            result = root(_func, initial_guess)

            if result.success:
                plotpred = result.x[0]
            else:
                plotpred = 0

            return round(plotpred)

        def _case_lower(plot, col_val, numerical_date_col, threshold):
            plot = plot.sort_values(
                numerical_date_col, ascending=True
            ).reset_index()
            x = plot[numerical_date_col].values
            y = plot[col_val].values

            spl = UnivariateSpline(x, y, k=3, s=4)
            # xs = np.linspace(x.min(), x.max(), 1000)

            def _func(x_val):
                return spl(x_val) - threshold

            initial_guess = x.astype(int).max()
            result = root(_func, initial_guess)

            if result.success:
                plotpred = result.x[0]
            else:
                plotpred = -997

            return round(plotpred)

        df1 = self.ldata.copy()
        plot_id_col = "id"
        col_val = band
        df1["num_day"] = (
            pd.to_datetime(df1.date) - pd.to_datetime(df1.date).min()
        )
        df1["num_day"] = (
            df1["num_day"].astype(str).apply(lambda x: int(x.split(" ")[0]))
        )
        numerical_date_col = "num_day"

        if from_day > 0:
            df1 = df1.loc[df1.num_day > from_day].copy()

        for p in df1[plot_id_col].unique():

            plot = df1.loc[df1[plot_id_col] == p]

            # First case if threshold is in rage
            if (plot[col_val].min() <= threshold) & (
                plot[col_val].values[: int((len(plot[col_val]) / 2))].max()
                >= threshold
            ):
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_in(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "IN"

            # Second case if threshold is upper than the range in col_val
            elif (
                plot[col_val].values[: int((len(plot[col_val]) / 2))].max()
                < threshold
            ):
                print(f"Plot Id: {p} threshold is upper than range ")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_upper(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "upper"

            # Third case if threshold is lower than the range in col_val
            elif plot[col_val].min() >= threshold:
                print(f"Plot Id: {p} threshold is lower than range ")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_lower(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "lower"

        if to_data:
            self.ldata = self.ldata.merge(
                df1.loc[
                    :,
                    [
                        "id",
                        numerical_date_col,
                        "dpred",
                        "in_range",
                    ],
                ],
                on=["id"],
                how="left",
            )
        else:
            return df1

    def get_senescens_Loess_predictions(
        self,
        band: str,
        threshold: float,
        frac_val=0.5,
        to_data: bool = False,
        from_day=0,
    ):
        """Generates predictions of senecense by providing threshold.

        Parameters
        ----------
        band:str
            Band name to be used in the prediciton.
        threshold:float
            value to determen if a plot is dry or not.
        to_data:bool
            boolean value to save or not the predictions.

        Returns
        -------
            Geodataframe
        """

        def _case_in(
            plot, col_val, numerical_date_col, threshold, frac_val=frac_val
        ):

            x = plot[numerical_date_col].values
            y = plot[col_val].values

            lowess_result = lowess(y, x, frac=frac_val)

            # Extract smoothed x and y
            x_smooth = lowess_result[:, 0]
            y_smooth = lowess_result[:, 1]

            # interpolation functions for prediction
            lowess_predict_x_from_y = interp1d(
                y_smooth, x_smooth, kind="linear", fill_value="extrapolate"
            )

            y_target = threshold
            x_for_y = lowess_predict_x_from_y(y_target)

            return int(np.round(x_for_y))

        def _case_upper(
            plot, col_val, numerical_date_col, threshold, frac_val=frac_val
        ):

            x = plot[numerical_date_col].values
            y = plot[col_val].values

            lowess_result = lowess(y, x, frac=frac_val)

            # Extract smoothed x and y
            x_smooth = lowess_result[:, 0]
            y_smooth = lowess_result[:, 1]

            # interpolation functions for prediction
            lowess_predict_x_from_y = interp1d(
                y_smooth, x_smooth, kind="linear", fill_value="extrapolate"
            )

            y_target = threshold
            x_for_y = lowess_predict_x_from_y(y_target)

            return int(np.round(x_for_y))

        def _case_lower(
            plot, col_val, numerical_date_col, threshold, frac_val=frac_val
        ):

            x = plot[numerical_date_col].values
            y = plot[col_val].values

            lowess_result = lowess(y, x, frac=frac_val)

            # Extract smoothed x and y
            x_smooth = lowess_result[:, 0]
            y_smooth = lowess_result[:, 1]

            # interpolation functions for prediction
            lowess_predict_x_from_y = interp1d(
                y_smooth, x_smooth, kind="linear", fill_value="extrapolate"
            )

            y_target = threshold
            x_for_y = lowess_predict_x_from_y(y_target)

            return int(np.round(x_for_y))

        df1 = self.ldata.copy()
        plot_id_col = "id"
        col_val = band
        df1["num_day"] = (
            pd.to_datetime(df1.date) - pd.to_datetime(df1.date).min()
        )
        df1["num_day"] = (
            df1["num_day"].astype(str).apply(lambda x: int(x.split(" ")[0]))
        )
        numerical_date_col = "num_day"

        if from_day > 0:
            df1 = df1.loc[df1.num_day > from_day].copy()

        for p in df1[plot_id_col].unique():

            plot = df1.loc[df1[plot_id_col] == p]

            # First case if threshold is in rage
            if (plot[col_val].min() <= threshold) & (
                plot[col_val].values[: int((len(plot[col_val]) / 2))].max()
                >= threshold
            ):
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_in(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "IN"

            # Second case if threshold is upper than the range in col_val
            # takes the first highes values and compares them.
            elif (
                plot[col_val].values[: int((len(plot[col_val]) / 2))].max()
                < threshold
            ):
                print(f"Plot Id: {p} threshold is upper than range ")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_upper(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "upper"

            # Third case if threshold is lower than the range in col_val
            elif plot[col_val].min() >= threshold:
                print(f"Plot Id: {p} threshold is lower than range")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_lower(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "lower"

        if to_data:
            self.ldata = self.ldata.merge(
                df1.loc[
                    :,
                    [
                        "id",
                        numerical_date_col,
                        "dpred",
                        "in_range",
                    ],
                ],
                on=["id"],
                how="left",
            )
        else:
            return df1
