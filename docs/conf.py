# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import sys


# add the spyctral source to the build path
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
PYNOMIC_PATH = CURRENT_PATH.parent

sys.path.insert(0, str(PYNOMIC_PATH))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pynomic'
copyright = '2025, Juan Manuel Fiore'
author = 'Juan Manuel Fiore'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx_mdinclude'
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = "index"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def setup(app):
    app.add_css_file("css/pynomic.css")
    app.add_js_file("js/pynomic.js")