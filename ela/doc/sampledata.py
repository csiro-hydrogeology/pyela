"""Material for easy access to sample data for pyela documentation

"""

import sys
import numpy as np
import pandas as pd
import os
import pkg_resources
import rasterio
import geopandas as gpd

DATA_PATH = pkg_resources.resource_filename('ela', 'data/')
SAMPLE_DATA_PATH = os.path.join(DATA_PATH, 'api_samples')

def sample_dem(folder=None):
    if folder is None: folder = SAMPLE_DATA_PATH
    sample_raster = rasterio.open(os.path.join(folder,'sample_dem.tif'))
    return sample_raster

def sample_bore_location(folder=None):
    if folder is None: folder = SAMPLE_DATA_PATH
    sample_bore_locations = gpd.read_file(os.path.join(folder,'bungendore_sample_logs.shp'))
    return sample_bore_locations

def sample_lithology_logs(folder=None):
    if folder is None: folder = SAMPLE_DATA_PATH
    lithology_logs = pd.read_csv(os.path.join(folder, 'litho_logs_sample.csv'))
    return lithology_logs

def sample_data(folder=None):
    return (sample_dem(folder), sample_bore_location(folder), sample_lithology_logs(folder))


DEPTH_FROM_COL = 'FromDepth'
DEPTH_TO_COL = 'ToDepth'

TOP_ELEV_COL = 'TopElev'
BOTTOM_ELEV_COL = 'BottomElev'

LITHO_DESC_COL = 'Description'
HYDRO_CODE_COL = 'HydroCode'

EASTING_COL = 'Easting'
NORTHING_COL = 'Northing'
