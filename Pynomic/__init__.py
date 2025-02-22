# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Pynomic library.
"""

# =============================================================================
# META
# =============================================================================

__version__ = "0.0.1"


# =============================================================================
# IMPORTS
# =============================================================================

from .core.core import Pynomicproject
from .core.plot import Pynomicplotter
from .io.get_plot_bands import read_zarr
from .io.get_plot_bands import process_stack_tiff

__all__ = ["Pynomicplotter", "Pynomicproject",
           "read_zarr", "process_stack_tiff"]
