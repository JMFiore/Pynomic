# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

"""Provides the functions for plotting."""

# =============================================================================
# IMPORTS
# =============================================================================
import attr

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np


# =============================================================================
# CLASSES
# =============================================================================


@attr.s(repr=False)
class Pynomicplotter:
    """class for ploting the data in Pynomicproject."""

    _summary = attr.ib()

    def __call__(self, kind="timeline", **kwargs):
        """Allows to give other functionalitys to the .plot."""
        method = getattr(self, kind)
        return method(**kwargs)

    def _set_common_labels(self, ax, title, obj_name, band, xlab):
        """Inner function to add titles and names to the axis."""
        ax.set_xlabel(xlab)
        ax.set_ylabel(band)
        ax.set_title(title + " - Plot Id " + obj_name)
        ax.grid(True)
        return

    def timeline(self, band_name, n_id, ax=None, days=False, **kwargs):
        """Plots a time line of a particular band.

        Parameters
        ----------
            band_name:str
                the name of the band in ldata to plot.
            n_id:str
                id of the plot.
            days:bool
                default False converts the x axis.

        Returns
        -------
            plot axis
        """
        df = self._summary.ldata

        plot = df.loc[df.id == n_id].copy()

        if days:
            plot["date"] = (
                pd.to_datetime(plot["date"])
                - pd.to_datetime(plot["date"]).min()
            )
            plot["date"] = plot["date"].dt.days
            xlab = "Flight days"
            rot = 0
        else:
            plot["date"] = pd.to_datetime(plot["date"]).dt.date
            xlab = "Flight dates"
            rot = 90

        plot = plot.sort_values("date").reset_index()
        plot.set_index("date", inplace=True)
        if ax is None:
            ax = plt.axes()

        ax.plot(plot.index, plot[band_name], **kwargs)
        ax.set_xticks(plot.index)
        ax.set_xticklabels(plot.index, rotation=rot)
        self._set_common_labels(ax, f"{band_name}", str(n_id), band_name, xlab)

        return ax

    def image_timeline(
        self,
        band_name,
        n_id,
        function,
        ax=None,
        days=False,
        vmin=0,
        vmax=255,
        **kwargs,
    ):
        """Generate a time line of images with line plot.

        Parameters
        ----------
            band_name:str
                the name of the band in ldata to plot.
            n_id:str
                id of the plot.
            days:bool
                default False converts the x axis.

            function:function
                a function that takes as input the array of the plots
                and returns an array to be ploted.

            vmin:int
                minimum value to be plotted.
            vmax:int
                maximium value to be plotted.

        Returns
        -------
            plot and axis
        """

        def _get_arrays(self, n_id, function):
            imlist = []
            dates = pd.DataFrame(self.ldata.date.unique(), columns=["date"])
            dates["datet"] = pd.to_datetime(dates["date"])
            dates = dates.loc[dates.datet.sort_values().index].reset_index()
            for dat in dates["date"].values:
                bands_names = []
                bands_arr = []
                for band in self.bands_name:
                    bands_names.append(band)
                    bands_arr.append(
                        self.raw_data["dates"][dat][n_id][band][:]
                    )
                imlist.append(function(dict(zip(bands_names, bands_arr))))
            return imlist

        array_list = _get_arrays(self._summary, n_id=n_id, function=function)
        df = self._summary.ldata
        plot = df.loc[df.id == n_id].copy()
        if days:
            plot["date"] = (
                pd.to_datetime(plot["date"])
                - pd.to_datetime(plot["date"]).min()
            )
            plot["date"] = plot["date"].dt.days
            xlab = "Flight days"
            rot = 0
        else:
            plot["date"] = pd.to_datetime(plot["date"]).dt.date
            xlab = "Flight dates"
            rot = 90

        fig = plt.figure()
        plt.suptitle(f"{band_name} - Plot Id {n_id}")
        gs = fig.add_gridspec(
            2, len(array_list), height_ratios=[1, 2], wspace=0.05, hspace=0.01
        )

        for impos, imval in enumerate(array_list):

            ax1 = fig.add_subplot(gs[0, impos])
            plt.imshow(imval, vmin=vmin, vmax=vmax)
            ax1.set_xticks([])
            ax1.set_yticks([])

        df = self._summary.ldata

        plot = df.loc[df.id == n_id].copy()
        if days:
            plot["date"] = (
                pd.to_datetime(plot["date"])
                - pd.to_datetime(plot["date"]).min()
            )
            plot["date"] = plot["date"].dt.days
            xlab = "Flight days"
            rot = 0
        else:
            plot["date"] = pd.to_datetime(plot["date"]).dt.date
            xlab = "Flight dates"
            rot = 90

        plot = plot.sort_values("date").reset_index()
        axbig = fig.add_subplot(gs[1, :])
        axbig = plt.gca() if axbig is None else axbig
        axbig.plot(plot.date, plot[band_name], **kwargs)
        axbig.set_xticks(plot.date)
        axbig.set_xticklabels(plot.date, rotation=rot)
        axbig.set_xlabel(xlab)
        axbig.set_ylabel(band_name)
        axbig.grid(True)

        return ax1, axbig

    def RGB_image_timeline(
        self,
        band_name,
        n_id,
        Red: str,
        Green: str,
        Blue: str,
        Size=(),
        ax=None,
        days=False,
        vmin=0,
        vmax=255,
        **kwargs,
    ):
        """Generate a time line of images with line plot showing a RGB function by default.

        Parameters
        ----------
            band_name:str
                the name of the band in ldata to plot.
            n_id:str
                id of the plot.
            Red:str
                red band name
            Green:str
                green band name
            Blue:str
                blue band name
            days:bool
                default False converts the x axis.

            function: a function that takes as input the array of the plots
                and returns an array to be ploted.

            vmin:int
                minimum value to be plotted.
            vmax:int
                maximium value to be plotted.

        Returns
        -------
            plot and axis
        """

        def _rgb_view(df, red1, blue1, green1, size: tuple):

            if len(size) > 0:

                up = size[0]
                down = size[1]
                left = size[2]
                right = size[3]

                red = df[red1][up:down, left:right]
                if (red.dtype == "float64") or ((red.dtype == "float32")):
                    red *= 255.0 / red.max()
                    red = np.uint8(red.astype(int))
                blue = df[blue1][up:down, left:right]
                if (blue.dtype == "float64") or ((blue.dtype == "float32")):
                    blue *= 255.0 / blue.max()
                    blue = np.uint8(blue.astype(int))
                green = df[green1][up:down, left:right]
                if (green.dtype == "float64") or ((green.dtype == "float32")):
                    green *= 255.0 / green.max()
                    green = np.uint8(green.astype(int))

            else:
                red = df[red1][:]
                if (red.dtype == "float64") or ((red.dtype == "float32")):
                    red *= 255.0 / red.max()
                    red = np.uint8(red.astype(int))
                blue = df[blue1][:]
                if (blue.dtype == "float64") or ((blue.dtype == "float32")):
                    blue *= 255.0 / blue.max()
                    blue = np.uint8(blue.astype(int))
                green = df[green1][:]
                if (green.dtype == "float64") or ((green.dtype == "float32")):
                    green *= 255.0 / green.max()
                    green = np.uint8(green.astype(int))

            image = np.dstack([red, green, blue])

            return image

        def _get_arrays(self, n_id, function):
            imlist = []
            dates = pd.DataFrame(self.ldata.date.unique(), columns=["date"])
            dates["datet"] = pd.to_datetime(dates["date"])
            dates = dates.loc[dates.datet.sort_values().index].reset_index()
            for dat in dates["date"].values:
                bands_names = []
                bands_arr = []
                for band in self.bands_name:
                    bands_names.append(band)
                    bands_arr.append(
                        self.raw_data["dates"][dat][n_id][band][:]
                    )
                imlist.append(
                    function(
                        dict(zip(bands_names, bands_arr)),
                        red1=Red,
                        green1=Green,
                        blue1=Blue,
                        size=Size,
                    )
                )
            return imlist

        array_list = _get_arrays(self._summary, n_id=n_id, function=_rgb_view)
        df = self._summary.ldata
        plot = df.loc[df.id == n_id].copy()
        if days:
            plot["date"] = (
                pd.to_datetime(plot["date"])
                - pd.to_datetime(plot["date"]).min()
            )
            plot["date"] = plot["date"].dt.days
            xlab = "Flight days"
            rot = 0
        else:
            plot["date"] = pd.to_datetime(plot["date"]).dt.date
            xlab = "Flight dates"
            rot = 90

        fig = plt.figure()
        plt.suptitle(f"{band_name} - Plot Id {n_id}")
        gs = fig.add_gridspec(
            2, len(array_list), height_ratios=[1, 2], wspace=0.05, hspace=0.01
        )

        for impos, imval in enumerate(array_list):

            ax1 = fig.add_subplot(gs[0, impos])
            plt.imshow(imval, vmin=vmin, vmax=vmax)
            ax1.set_xticks([])
            ax1.set_yticks([])

        df = self._summary.ldata

        plot = df.loc[df.id == n_id].copy()
        if days:
            plot["date"] = (
                pd.to_datetime(plot["date"])
                - pd.to_datetime(plot["date"]).min()
            )
            plot["date"] = plot["date"].dt.days
            xlab = "Flight days"
            rot = 0
        else:
            plot["date"] = pd.to_datetime(plot["date"]).dt.date
            xlab = "Flight dates"
            rot = 90

        plot = plot.sort_values("date").reset_index()
        axbig = fig.add_subplot(gs[1, :])
        axbig = plt.gca() if axbig is None else axbig
        axbig.plot(plot.date, plot[band_name], **kwargs)
        axbig.set_xticks(plot.date)
        axbig.set_xticklabels(plot.date, rotation=rot)
        axbig.set_xlabel(xlab)
        axbig.set_ylabel(band_name)
        axbig.grid(True)

        return ax1, axbig
