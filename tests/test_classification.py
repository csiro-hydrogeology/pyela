import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.append(pkg_dir)

from ela.classification import *

def test_numeric_code_mapping():
    lithologies = ['sand','clay','silt']
    codes = create_numeric_classes(lithologies)
    assert len(codes) == 3
    assert codes['sand'] == 0.0
    assert codes['clay'] == 1.0
    assert codes['silt'] == 2.0
    test_lithologies = ['silt','sand','silt','sand','clay','silt']
    vcodes = v_to_litho_class_num(test_lithologies, codes)
    assert len(vcodes) == 6
    assert vcodes[0] == 2.0
    assert vcodes[1] == 0.0
    assert vcodes[4] == 1.0

def test_get_lithology_observations_for_depth():
    obs_colname = 'fake_obs'
    slice_depth = 5.0
    mock_obs = pd.DataFrame(
    {EASTING_COL:np.array([.0, 1., 1., 0., 0.]), 
    NORTHING_COL:np.array([2., 2., 3., 3., 12.]), 
    DEPTH_TO_AHD_COL:np.array([2., 2., 3., 3., 3.]),
    DEPTH_FROM_AHD_COL:np.array([6., 4., 5., 4., 10.]),
    obs_colname: np.array([.1, .2, .3, .4, np.nan])
    })
    df = get_lithology_observations_for_depth(mock_obs, slice_depth, obs_colname)
    # only the first and third row match. The last one has missing observation (nan) so excluded
    assert len(df) == 2
    xx, yy, zz_from, zz_to, ss = extract_bore_class_num(df, obs_colname)
    assert xx[0] == .0
    assert xx[1] == 1.
    assert yy[0] == 2.
    assert yy[1] == 3.
    assert zz_from[0] == 6.
    assert zz_from[1] == 5.
    assert zz_to[0] == 2.
    assert zz_to[1] == 3.
    assert ss[0] == .1
    assert ss[1] == .3

# test_get_lithology_observations_for_depth()
