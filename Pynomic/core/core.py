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
import io
import os
from PIL import Image

# =============================================================================
# CLASSES
# =============================================================================


@attrs.define
class Pynomicproject:
    """Contains all the extracted bands from each plot and dates.

    Attributes
    ----------
    raw_data : dict-like
        contains all the data.

    ldata : Pandas Dataframe
        contains all the procesed data.
    """

    raw_data: zarr.hierarchy
    ldata: pd.DataFrame
    n_dates: int
    dates: list
    n_bands: int
    bands_name: list

    def __getitem__(self, k: str):
        """Allow attribute access using dictionary-like syntax.

        Parameters
        ----------
        k : str
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
        """Calculates Vegetation index.

        Args
            Red: name of the column that contains the red band
            Blue: name of the column that contains the blue band
            Green: name of the column that contains the green band

        Returns
        -------
            Vegetation index in the ldata object.
        """
        df = self.ldata
        red = df.loc[:, Red]
        blue = df.loc[:, Blue]
        green = df.loc[:, Green]

        df["VDVI"] = (2 * green - red - blue) / (
            2 * green + red + blue
        )  # Visible-band difference vegetation index
        df["NGRDI"] = (green - red) / (
            green + red
        )  # Normalized green–red difference index (Kawashima Index)
        df["VARI"] = (green - red) / (
            green + red - blue
        )  # Visible Atmospherically Resistant Index
        df["GRRI"] = green / red  # Green–red ratio index
        df["VEG"] = green / (
            (red**0.667) * (blue ** (1 - 0.667))
        )  # Vegetativen
        df["MGRVI"] = ((green**2) - (red**2)) / (
            (green**2) + (blue**2)
        )  # Modified Green Red Vegetation Index
        df["GLI"] = (2 * green - red - blue) / (
            (-red) - blue
        )  # Green Leaf Index
        df["ExR"] = (1.4 * red - green) / (
            green + red + blue
        )  # Excess Red Vegetation Index
        df["ExB"] = (1.4 * blue - green) / (
            green + red + blue
        )  # Excess Blue Vegetation Index
        df["ExG"] = 2 * green - red - blue  # Excess Green Vegetation Index
        return

    def generate_unique_feature(
        self, function, features_names: list, to_data=False
    ):
        """Higher order function that iterate through the flight dates.

        Parameters
        ----------
        function : a function that contains a formula and
        returns a sigle value

        new_name : str
                    the name of the new feature.

        to_data : bool
                merges it with the project data.

        Returns
        -------
            dataframe with the new index.
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
                df.id = df.id.astype(int)
                self.ldata = self.ldata.merge(df, on=["id", "date"])
                return self.ldata

            else:

                return pd.DataFrame(values_list, columns=features_names)
        else:
            return print("feature_names is not a list")

    def get_senescens_predictions(
        self, band: str, threshold: float, to_data: bool = False, from_day = 0
    ):
        """Generates predictions of senecense by providing threshold and index.

        Args
            band: Band name to be used in the prediciton.
            threshold: value to determen if a plot is dry or not.
            to_data: boolean value to save or not the predictions.

        Returns
        -------
            Dataframe
        """

        def _case_in(plot, col_val, numerical_date_col, threshold ):

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

            # filtro para quedarme con los valores negativos que se que
            # corresponden a valores iniciales.
            # menores al threshold pr lo tanto me van a dar valores negativos.
            # pero no con una adecuada pendiente.
            # por lo tanto erronea. Ajusto modelo para que tome el primer
            # valor de la serie y el mas bajo.
            ant_date = plot[numerical_date_col].values[0]
            colant_val = plot.loc[
                plot[numerical_date_col] == ant_date, col_val
            ].values[0]

            col_value = plot[col_val].min()
            plotval = plot.loc[
                plot[col_val] == col_value, numerical_date_col
            ].values[0]
            yval = np.array([ant_date, plotval]).reshape(-1, 1)
            xval = np.array([colant_val, col_value]).reshape(-1, 1)
            lm = LinearRegression().fit(xval, yval)
            plotpred = lm.predict(np.array([threshold]).reshape(-1, 1))[0][0]

            return round(plotpred)

        def _case_lower(plot, col_val, numerical_date_col, threshold):
            # Function to predict cases where the times series does not
            # reach the threshold because its to low.(next implement segreg)
            ant_date = plot[numerical_date_col].values[0]
            colant_val = plot.loc[
                plot[numerical_date_col] == ant_date, col_val
            ].values[0]

            col_value = plot[col_val].values[len(plot[col_val]) - 1]
            plotval = plot.loc[
                plot[col_val] == col_value, numerical_date_col
            ].values[0]

            yval = np.array([ant_date, plotval]).reshape(-1, 1)
            xval = np.array([colant_val, col_value]).reshape(-1, 1)
            lm = LinearRegression().fit(xval, yval)
            plotpred = lm.predict(np.array([threshold]).reshape(-1, 1))[0][0]

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

        if from_day > 0 :
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
                print(f"Plot Id: {p} range is lower than threshold ")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_upper(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "lower"

            # Third case if threshold is lower than the range in col_val
            elif plot[col_val].min() >= threshold:
                print(f"Plot Id: {p} range is Higher than threshold ")
                df1.loc[df1[plot_id_col] == p, "dpred"] = _case_lower(
                    plot, col_val, numerical_date_col, threshold
                )
                df1.loc[df1[plot_id_col] == p, "in_range"] = "upper"

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
                how = 'left'
            )
        else:
            return df1

    @property
    def plot(self):
        """Generate plots from spectra."""
        from .plot import Pynomicplotter

        return Pynomicplotter(self)

    def save(self, path):
        """Function to save project as .zip file.

        Args
            path: Name of the file ending with .zip.

        Returns
        -------
            a zipped folder in the path given.
        """
        if "bands_name" not in list(self.raw_data.array_keys()):
            self.raw_data.create_group("bands_name")
            self.raw_data["bands_name"] = self.bands_name

            df_buffer = io.BytesIO()
            self.ldata.to_parquet(df_buffer, engine="pyarrow")
            self.raw_data.create_group("ldata")
            self.raw_data["ldata"] = [df_buffer.getbuffer().tobytes()]

            store = zarr.ZipStore(path, mode="w")
            zarr.copy_store(self.raw_data.store, store)
            store.close()
            return

        else:
            self.raw_data["bands_name"] = self.bands_name

            df_buffer = io.BytesIO()
            self.ldata.to_parquet(df_buffer, engine="pyarrow")
            self.raw_data["ldata"] = [df_buffer.getbuffer().tobytes()]

            store = zarr.ZipStore(path, mode="w")
            zarr.copy_store(self.raw_data.store, store)
            store.close()
            return

    def save_plots_as_tiff(self, folder_path, fun, identification_col):
        """Creates as many folders as dates in path provided and saves the plot images.

        Args:
            folder_path: Path where to save the images.
            fun: function to use to stack the bands.
            identification_col: Column of ldata where the ids are.

        Returns
        -------
            folder with images
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
                name = str(self.ldata.loc[
                    self.ldata["id"] == int(p), identification_col
                ].unique()[0])
                image_path = os.path.join(path, name + ".tiff")
                image = Image.fromarray(arrays)
                image.save(image_path)
