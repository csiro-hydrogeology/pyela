import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__),'../..')

sys.path.insert(0, pkg_dir)

from ela.doc.sampledata import *

# get_coords_from_gpd_shape

from ela.spatial import get_coords_from_gpd_shape
dem ,bore_loc, litho_logs = sample_data()
df = litho_logs[[DEPTH_FROM_COL, DEPTH_TO_COL,TOP_ELEV_COL,BOTTOM_ELEV_COL,LITHO_DESC_COL,HYDRO_CODE_COL]]
geoloc = get_coords_from_gpd_shape(bore_loc, colname='geometry', out_colnames=[EASTING_COL, NORTHING_COL])
geoloc[HYDRO_CODE_COL] = bore_loc[HYDRO_CODE_COL]
geoloc.head()
# With this data frame we can perform two operations in one go: subsetting the lithology records to only the 640 bores of interest, and adding to the result the x/y geolocations to the data frame.
df = pd.merge(df, geoloc, how='inner', on=HYDRO_CODE_COL, sort=False, copy=True, indicator=False, validate=None)
