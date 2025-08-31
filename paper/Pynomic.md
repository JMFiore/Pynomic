

# Pynomic: Data extraction image library for orthomosaic breeding trials

**Abstract**

Pynomic, a Python-based package, was developed to automate the
extraction, visualization, and analysis of data from time-series
orthomosaics for plant breeding trials, addressing challenges in data
management and computational load. The increasing complexity and volume
of orthomosaic data, derived from multiple acquisition dates for
high-throughput phenotyping (HTP), often result in tedious, error-prone,
and non-scalable extraction processes with existing workflows. Pynomic
automates plot-level data extraction and facilitates time-series
visualization for monitoring growth, senescence, and stress patterns.
Furthermore, it allows the incorporation of senescence models and
comparison with plot-level features.

Pynomic\'s optimized data storage approach reduces disk space and Random
Access Memory (RAM) requirements, thereby decreasing computational
demands compared to traditional methods. While many existing workflows
offer partial solutions, Pynomic provides an integrated and customizable
pipeline. This flexibility, which includes a pre-built pipeline and the
ability to integrate new analyses or models from the Python ecosystem,
sets Pynomic apart. Consequently, Pynomic reduces the computational
burden associated with large-scale orthomosaic processing, facilitating
more extensive temporal analyses without requiring high-end computing
infrastructure. As a result, Pynomic\'s ability to simplify data
management and reduce resource requirements makes advanced phenotyping
more accessible and efficient, fostering reproducibility in plant
breeding research.

**Highlights**

1.  Automated Time-Series Phenotyping: Pynomic automates the extraction
    of plot-level data from time-series UAV orthomosaics (both RGB and
    Multispectral), streamlining high-throughput phenotyping workflows.

2.  Optimized Computational Efficiency: Significantly reduces RAM usage
    and disk space requirements for large datasets through its
    Zarr-based backend and lazy loading capabilities, leading to faster
    processing times.

3.  Flexible Feature Engineering: Offers built-in calculation of common
    Vegetation Indices (VIs) and Textural Features (TFs), alongside a
    framework for users to easily define and apply custom feature
    extraction functions.

4.  Integrated Analytical Tools: Includes functionalities for practical
    applications, such as senescence and maturity prediction models, and
    facilitates seamless integration with the broader Python machine
    learning ecosystem.

5.  Open-Source and Quality-Focused: Provides a user-friendly,
    well-documented, and rigorously tested open-source library,
    developed to high standards to foster reproducibility and
    accessibility in plant phenotyping research.

**Introduction**

The use of Unmanned Aerial Vehicles (UAVs) reduces inspection time by
covering large areas in minutes and decreasing crop management costs
(Shakoor et al., 2017). Their use is often facilitated by user-friendly,
ready-to-fly interfaces developed by commercial enterprises. The
generation of precise, rapid, objective, and less expensive information
compared to traditional phenotyping methods is the primary reason why
high-throughput phenotyping (HTP) with UAVs is being widely adopted by
breeding organizations. The adequate use of this technology, which
enables accurate and rapid screening of thousands of plots in
multi-location field trials, is of utmost importance for decision-makers
to improve breeding strategies (Shakoor et al., 2017). The number of
researchers focusing on HTP has shown an upward trend in recent years,
highlighting the potential and applications of UAVs in the agricultural
sector (Shakoor et al., 2019). Phenotyping not only allows for the
selection of the best individuals in the breeding process but, by
incorporating genomic and climatic data into machine learning and
statistical models, also improves our understanding of how each genotype
reacts to its environment (Samaras et al., 2019).

UAVs produce massive amounts of data that must be processed correctly to
conduct timely analyses (Guimarães et al., 2020). This data volume
increases further when analyses span multiple time points, necessitating
robust methods for managing large datasets containing both spatial and
temporal information (Nabwire et al., 2021). While the use of Red,
Green, and Blue (RGB) images is common, the trend towards using
multispectral sensors on UAVs, which enrich data analysis, is
increasing; these can offer a more comprehensive data solution,
sometimes more cost-effectively in the long term than RGB alone. The use
of other sensor types, such as hyperspectral sensors, is also on the
rise (Jang et al., 2020). Single temporal images can be employed for
certain analyses, such as plant count, stand assessment, or lodging (Koh
et al., 2021; Lu et al., 2023). However, for more complex analyses like
yield prediction or maturity estimation, high spatio-temporal resolution
is crucial, often necessitating more frequent revisits (Yuan et al.,
2024). This, in turn, comes at the cost of increased flight time and
data volume. Finding the optimal balance between spatial and temporal
resolution remains a key parameter to be defined for specific
applications (Samaras et al., 2019; Volpato et al., 2021).

Data extraction for phenotyping can be approached from two main
perspectives, each with implications for managing the aforementioned
data challenges. On one hand, feature engineering relies heavily on
researcher criteria for feature creation, a process that can be
laborious and time-consuming. In image analysis, this typically involves
calculating the mean of each band and then deriving Vegetation Indices
(VIs). This is the most commonly used approach. Alternatively, Textural
Features (TFs) can be extracted. This may involve retaining the original
pixel matrix for each plot and band, or generating VI matrices from band
combinations, and then applying algorithms like the Gray-Level
Co-occurrence Matrix (GLCM) to derive textural information. Combining
multiple VIs and TFs from various flight dates has been shown to improve
prediction accuracy (Ren et al., 2023; Wang et al., 2021; Zeng et al.,
2021; Zheng et al., 2019). On the other hand, automatic feature
extraction and generation involves feeding the image as input data,
allowing the model to autonomously extract and select the necessary
information. This approach is commonly implemented using Deep Learning
(DL) models (Moeinizade et al., 2022). However, drawbacks of DL models
include the significantly larger volumes of training data required,
increased model complexity, and higher demand for computational
resources. In such cases, ensuring that images and arrays are easily
manageable and traceable is critical.

The present work introduces a new open-source tool called Pynomic,
designed for data extraction from orthomosaics generated by UAVs in a
time-series format for plot trials. This tool allows for subsequent
integration with the Python machine learning and data analysis library
ecosystem.

**2. Pynomic: A Python Library for Orthomosaic Time-Series Analysis**

The primary goal of the Pynomic library is to automate the processing
pipeline from raw orthomosaics, such as those generated by software like
OpenDroneMap ([ODM](https://www.opendronemap.org), 2020) and [Pix4D](http://www.pix4d.com)
to a structured GeoPandasDataframe (GPDF). This GPDF contains extracted features
for each area of interest (AOI) in this context, experimental
plots organized by date. It's core data
structure is based on the Zarr hierarchy (specifically, zarr.Group
objects) from the Zarr library (Zarr Development Team, 2024). The
process begins by cropping each AOI based on provided geometries,
determining its vertical position and storing the pixel data within this Zarr
hierarchy object. This structure enables efficient \"lazy loading,\" allowing
large collections of AOI data to be stored on disk and accessed rapidly
on demand, thereby significantly reducing Random Access Memory (RAM)
usage. This approach also optimizes the storage of essential
information, minimizing the overall disk space required for a project.

**2.1 The PynomicProject Object**

The PynomicProject object (hereafter PO) is the central class that
encapsulates all information from a given experimental trial. The PO
stores the raw pixel matrices for each band and date for every plot,
typically within its *PO.raw_data* attribute (a Zarr group). Extracted
features and metadata are stored in *PO.ldata*, a GeoPandas DataFrame in
a \"long\" format, which also includes relevant information such as band
names, acquisition dates, and plot identifiers.


**2.2 Workflow**

Initiating the data processing is done by calling  the function
*PynomicProject.from_tiff_stack()*. This function automatically
processes a series of input TIFF orthomosaics, extracts data for each
plot and band according to a provided grid file, and stores it within
the PO. By default, this initial processing might calculate and store
the mean of each band per plot.

To enhance the analysis, users can then employ various methods of the
PO. For instance:

- *.calculate_rgb_vi()*: Calculates common vegetation indices (VIs) from
  RGB bands.

- *.calculate_multispectral_vi()*: Calculates VIs suitable for
  multispectral data (e.g., NDVI, EVI, SAVI).

- *.calculate_glcm_textures()*: Computes textural features using
  Gray-Level Co-occurrence Matrix (GLCM) methods for specified bands or
  VIs.

For more specialized feature extraction, the
*.generate_custom_feature()* method accepts a user-defined function.
This custom function typically receives a dictionary containing all band
matrices for a single plot at a single time point. Pynomic then
automatically applies this function to all plots across all dates, and
the derived features are appended to the *PO.ldata* DataFrame.

With features generated, users can proceed to tasks such as:

- Predicting phenological stages (e.g., maturity date) using methods
  like *.predict_senescence()* (which might incorporate models like
  splines or LOESS as demonstrated in the use case).

- Training custom machine learning models for outcomes like yield or
  flowering date, using the features in *PO.ldata*.

The suitability of features for predicting a response variable can be
interactively assessed using the *PO.plot_RGB_image()* method. This
visualizes time-series images of a plot alongside the temporal profile
of a selected feature.

Finally, the entire project (including raw data snippets and extracted
features) can be saved for later use with a method like
*PO.save_project(\'my_project.zarr\').* Alternatively, if the primary
goal is to train deep learning models, Pynomic can export the cropped,
oriented plot images in a format ready for model ingestion. The feature
DataFrame can also be exported (e.g.,
*PO.ldata.to_csv(\'features.csv\')*).


**2.3. Quality Standards**

To ensure code reliability and correctness, comprehensive unit tests
have been implemented across all major functions and methods using a
framework like Pytest. Each function has its own tests to validate
inputs, outputs, and internal logic, ensuring the correctness of output
values and data types. Current test coverage exceeds 95% of the
codebase. The library adheres to PEP 8 coding standards, promoting a
consistent style that improves code interpretation, scalability,
maintainability, and collaboration. Comprehensive [documentation](https://pynomic.readthedocs.io/en/latest/),
including tutorials and API references, has been created to help users
effectively integrate Pynomic into their research projects.

**2.4. Use Case: Maturity Prediction in Soybean and Wheat**

In this example, we demonstrate Pynomic\'s application for maturity
prediction using open-source datasets: a wheat trial with multispectral
data (Matias et al., 2022) and a soybean trial with RGB data (Lorenz,
2020). For senescence detection and maturity prediction from time-series
VI data, Pynomic can integrate or facilitate the use of curve-fitting
methods like LOESS or spline functions.
```
import Pynomic

from sklearn.metrics import root_mean_square_error

# process_tiff_stack requires 4 inputs the folder which 
# contains all the orthomisaics from the location starting
# with the date eg:
# 20250115_location.tiff. The grid that can be geojson or
# shape format.
# A unique id for each plot and a list of the bands in order
# orthomisaics from the location starting with the date eg:*
# 20250115_location.tiff. The grid that can be
# geojson or shape format.
# A unique id for each plot and a list of the bands in order.

wheat = Pynomic.porcess_tiff_stack(
folder="folder_orthomosaics",
grid_path="trial_grid.shp",
Id = 'Trial_number',
bands_name =['red','green','blue','nir','red_edge']
)

wheat.Multispectral_VI() # Generates the features

wheat.image_timeseries(band='NDVI',
id = "A1",
fun = rgb_func,
days = True
) # Generate a time series plot to compare the
# feature and images. In this case we can obsevre
# the near infra-red (NIR) band through time for
# the plot with the id = A1.
```
![alt text](output-1.png)

```

wheat.predict_Splines_Senescence(band = 'VDVI',
threshold = 0.1,
to_data=True) # Choosing the band and the
# threshold to make the predictions.

root_mean_square_error(
  wheat.ldata.MaturityDay.values,
  wheat.ldata.predictions.values) ## RSME 1.89 days.

# Finally we can save the project with

wheat.save('wheat_Maturity_project.zarr')

# or save the geopandas dataframe as csv

what.ldata.to_csv('Wheat_Maturity_predictions.csv')
```

A similar process was applied to an RGB dataset for soybean with the
exception of using the *.calculate_rgb_vi()* method and focusing on an
RGB-based VI like VDVI (with a threshold of 0.12) for predictions. The
predict_senescence method can add columns to the DataFrame indicating
prediction reliability (e.g., whether the senescence threshold was
crossed within the observed flight dates) and the predicted number of
days to senescence from the first flight. For instance, in the soybean
trial, predictions for plots where senescence occurred within the
observation period achieved a Root Mean Square Error (RMSE) of
approximately 2.19 days.

> **3. Conclusions**

The research presented culminates in the release of Pynomic version 1.0,
an open-source Python library engineered to streamline and automate the
complex data pipeline in high-throughput plant phenotyping. Pynomic
offers robust functionality for both RGB and multispectral UAV-derived
datasets, automating plot-level data extraction and feature generation.
Its design, leveraging an efficient Zarr-based backend, substantially
reduces processing time and mitigates computational burdens,
particularly in terms of RAM usage and disk storage.

Pynomic has been developed with a commitment to quality, adhering to
enterprise coding standards to ensure it is readily understandable,
scalable, and maintainable. Comprehensive user documentation and
integrated functionalities, such as models for maturity prediction,
further establish Pynomic as a practical and valuable tool. By
simplifying data handling and analysis, Pynomic is poised to support
ongoing research and facilitate the day-to-day work involved in
UAV-based plant phenotyping. It is our hope that this contribution will
encourage continued research and expanded application of HTP
technologies within the plant science community
