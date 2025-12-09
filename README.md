# Pynomic

![logo](https://raw.githubusercontent.com/JMFiore/Pynomic/master/docs/_static/pynomic_logo.svg)


![Tests](https://github.com/JMFiore/Pynomic/actions/workflows/testing.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pynomic/badge/?version=latest)](https://pynomic.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pynomic?period=total&units=ABBREVIATION&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pynomic)


*Pynomic* is a Python library designed to quickly and efficiently extract data from aerial images of plant breeding trials, offering a simple, automated, and trackable solution.

## Motivation
The use of drones in agriculture is beeing widely spread due to its versatility from plant treatment aplication, sowing, scouting crops, etc. In plant breeding it allows to capture large amounts of Phenomic data with less human effort.
This data needs to be processed in order to extract usefull information. Hence Pynomic the tool to solve this porblem. It combines librarys to get the best of them, making it able to fast extract the data, plot each treatment and it's evolution
through time, estimate the date of a event given a certain threshold, automaticaly calculate RGB and Multispectra vegetation indices while at the same time reducing the need for RAM. Storing only the relevant information. Thus reducing the computation resources.

## Requirements
You need Python 3.9+ to run Pynomic.

## Basic install
You can find Pynomic in PyPl. The standard installation.

    $ pip install Pynomic


## Developer Installation 
Clone this repo. Move to the file and execute:

    $ pip install e .

