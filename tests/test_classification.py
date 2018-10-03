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

def test_make_training_set():
    obs_colname = 'fake_obs'
    mock_obs = pd.DataFrame({EASTING_COL:np.array([.0, 1., 1., 0.]), NORTHING_COL:np.array([2., 2., 3., 3.]), obs_colname: np.array([.1, .2, .3, .4])})
    X, y = make_training_set(mock_obs, obs_colname)
    assert X.shape[0] == 4
    assert X.shape[1] == 2
    assert y.shape[0] == 4

def mock_litho_drill(x, y, obs_colname = 'fake_obs'):
    z_min = -10
    s = x+y-z_min
    zz = range(z_min,0,2)
    tops = [z+2 for z in zz]
    bottoms = [z for z in zz]
    obs_val = [(z+s) for z in zz]
    mock_obs = pd.DataFrame({
        EASTING_COL:np.array([x for z in zz]), 
        NORTHING_COL:np.array([y for z in zz]), 
        DEPTH_FROM_AHD_COL:np.array(tops),
        DEPTH_TO_AHD_COL:np.array(bottoms),
        obs_colname: np.array(obs_val, dtype='float64')
    })
    return mock_obs
    #>>> mock_litho_drill(1,1)
    #   Easting  Northing  Depth To (AHD)  Depth From (AHD)  fake_obs
    #0        1         1              -8               -10       2.0
    #1        1         1              -6                -8       4.0
    #2        1         1              -4                -6       6.0
    #3        1         1              -2                -4       8.0
    #4        1         1               0                -2      10.0

def mock_litho_drill_grid(dim_x = 10, dim_y = 20, obs_colname = 'fake_obs'):
    dframes = [mock_litho_drill(i,j, obs_colname) for i in range(dim_x) for j in range(dim_y)]
    return pd.concat(dframes)

def test_get_knn_model():
    obs_colname = 'fake_obs'
    dim_x = 10
    dim_y = 20
    df = mock_litho_drill_grid(dim_x , dim_y, obs_colname)
    slice_depth = -5.0
    n_neighbours = 1000
    knn_trained = get_knn_model(df, obs_colname, slice_depth, n_neighbours)
    assert knn_trained is None
    n_neighbours = 6
    knn_trained = get_knn_model(df, obs_colname, slice_depth, n_neighbours)
    assert knn_trained is not None

def test_class_probability_estimates_depth():
    obs_colname = 'fake_obs'
    dim_x = 2
    dim_y = 3
    df = mock_litho_drill_grid(dim_x , dim_y, obs_colname)
    slice_depth = -1
    n_neighbours = 4
    all_classes = [klass for klass in set(df[obs_colname])]
    pad_training_set = pad_training_set_functor(all_classes)
    mesh_grid = create_meshgrid_cartesian(x_min=0, x_max=4, y_min=2, y_max=5, grid_res=0.99)
    probas = class_probability_estimates_depth(df, obs_colname, slice_depth, n_neighbours, mesh_grid, func_training_set=None)
    assert len(probas) < len(all_classes)
    probas = class_probability_estimates_depth(df, obs_colname, slice_depth, n_neighbours, mesh_grid, func_training_set=pad_training_set)
    assert len(probas) == len(all_classes)

