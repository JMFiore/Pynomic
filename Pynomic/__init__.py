# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# Copyright (c) 2024, Fiore J.Manuel.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Pynomic.

Implementation drone images analisys.
"""

# =============================================================================
# META
# =============================================================================

__version__ = "0.0.1"


# =============================================================================
# IMPORTS
# =============================================================================

from .core import Pynomicproject
from .io.get_plot_bands import process_stack_tiff
from .io.get_plot_bands import auto_fit_image
from .io.get_plot_bands import read_zarr

__all__ = ["Pynomicproject", "process_stack_tiff",
           "auto_fit_image", "read_zarr"]
