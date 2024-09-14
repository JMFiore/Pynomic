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

    raw_data: dict
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

    def generate_unique_feature(self, function, new_name: str, to_data=False):
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
        raw_data = self.raw_data
        values_dict = {"id": [], "date": [], "old_name": []}

        for flight_dates in raw_data.keys():
            for plot in raw_data[flight_dates].keys():
                bands = raw_data[flight_dates][plot]
                values_dict["date"].append(flight_dates.split("_")[0])
                values_dict["id"].append(plot)
                values_dict["old_name"].append(function(bands))
        values_dict[new_name] = values_dict.pop("old_name")

        if to_data:

            df = pd.DataFrame(values_dict)
            self.ldata = self.ldata.merge(df, on=["id", "date"])
            return

        else:

            return pd.DataFrame(values_dict)
