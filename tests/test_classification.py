import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.insert(0, pkg_dir)

from ela.classification import *

def test_numeric_code_mapping():
    lithologies = ['sand','clay','silt']
    codes = create_numeric_classes(lithologies)
    assert len(codes) == 3
    assert codes['sand'] == 0.0
    assert codes['clay'] == 1.0
    assert codes['silt'] == 2.0
    test_lithologies = ['silt','sand','silt','sand','clay','silt','magma']
    vcodes = v_to_litho_class_num(test_lithologies, codes)
    assert len(vcodes) == 7
    assert vcodes[0] == 2.0
    assert vcodes[1] == 0.0
    assert vcodes[4] == 1.0
    assert np.isnan(vcodes[6])

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
    dfcn = GeospatialDataFrameColumnNames(EASTING_COL, NORTHING_COL, DEPTH_FROM_AHD_COL, DEPTH_TO_AHD_COL)
    df = dfcn.get_lithology_observations_for_depth(mock_obs, slice_depth, obs_colname)
    # only the first and third row match. The last one has missing observation (nan) so excluded
    assert len(df) == 2
    xx, yy, zz_from, zz_to, ss = dfcn.extract_bore_class_num(df, obs_colname)
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
    dfcn = GeospatialDataFrameColumnNames(EASTING_COL, NORTHING_COL, DEPTH_FROM_AHD_COL, DEPTH_TO_AHD_COL)
    X, y = dfcn.make_training_set(mock_obs, obs_colname)
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
    dfcn = GeospatialDataFrameColumnNames()
    knn_trained = dfcn.get_knn_model(df, obs_colname, slice_depth, n_neighbours)
    assert knn_trained is None
    n_neighbours = 6
    knn_trained = dfcn.get_knn_model(df, obs_colname, slice_depth, n_neighbours)
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
    dfcn = GeospatialDataFrameColumnNames()
    probas = dfcn.class_probability_estimates_depth(df, obs_colname, slice_depth, n_neighbours, mesh_grid, func_training_set=None)
    assert len(probas) < len(all_classes)
    probas = dfcn.class_probability_estimates_depth(df, obs_colname, slice_depth, n_neighbours, mesh_grid, func_training_set=pad_training_set)
    assert len(probas) == len(all_classes)

def test_class_mapping():
    lithology_names = ['sand','clay','limestone']
    sand_code = 0
    clay_code = 1
    limestone_code = 2
    fast = 0
    medium = 1
    slow = 2
    mapping = {
        'sand/':fast,
        'sand/sand':fast,
        'sand/clay':fast,
        'sand/limestone':fast,
        'clay/':slow,
        'clay/clay':slow,
        'clay/sand':medium,
        'clay/limestone':medium,
        'limestone/':fast,
        'limestone/limestone':fast,
        'limestone/sand':fast,
        'limestone/clay':medium,
    }
    mapper = ClassMapper(mapping, lithology_names)

    assert mapper.litho_class_label( sand_code, clay_code ) == 'sand/clay'
    assert mapper.litho_class_label( sand_code, 'clay' ) == 'sand/clay'
    assert mapper.litho_class_label( sand_code, np.nan ) == 'sand/'
    assert mapper.litho_class_label( 'sand', np.nan ) == 'sand/'
    assert mapper.litho_class_label( 'unknown_thing', 'clay' ) == ''
    assert mapper.litho_class_label( 'unknown_thing', np.nan ) == ''
    assert mapper.litho_class_label( np.nan, np.nan ) == ''

    assert mapper.class_code('sand','')  == fast
    assert mapper.class_code('sand','sand')  == fast
    assert mapper.class_code('sand',np.nan)  == fast
    assert mapper.class_code(sand_code,sand_code)  == fast
    assert mapper.class_code(sand_code,'')  == fast
    assert mapper.class_code(clay_code,np.nan)  == slow
    assert np.isnan(mapper.class_code('unknown_thing',np.nan))

    assert mapper.bivariate_mapper(sand_code,sand_code)  == fast
    assert mapper.bivariate_mapper(sand_code,np.nan)  == fast
    assert mapper.bivariate_mapper(clay_code,np.nan)  == slow
    assert np.isnan(mapper.bivariate_mapper(np.nan,np.nan))
    assert np.isnan(mapper.bivariate_mapper(np.nan,clay_code))

    num_remap = mapper.numeric_for_litho_classes(['sand/clay', 'clay/sand'])
    assert num_remap[0] == fast
    assert num_remap[1] == medium


    primary_lithology_3d_array = np.empty([3,4,5])
    secondary_lithology_3d_array = np.empty([3,4,5])
    mask_2d = np.empty([3,4], dtype=np.bool)
    mask_2d[:,:] = True
    primary_lithology_3d_array[:,:,:] = float(sand_code)
    secondary_lithology_3d_array[:,:,:] = float(limestone_code)
    primary_lithology_3d_array[0,:,:] = float(clay_code)
    secondary_lithology_3d_array[:,:,0] = float(clay_code)
    secondary_lithology_3d_array[2,2,:] = np.nan
    mask_2d[0,:] = False
    freq_table = mapper.get_frequencies(mask_2d, primary_lithology_3d_array, secondary_lithology_3d_array)
    assert freq_table[sand_code, sand_code] == 5 # where secondary_lithology_3d_array is np.nan
    assert freq_table[sand_code, clay_code] == (3-1) * 4 * (5-4) - 1 # lowest 2ndary slice all clay, but x=0 is out of the mask, and one cell x=2,y=2 is nan
    assert freq_table[sand_code, limestone_code] == ((3-1) * 4 * (5-1) - 4*1) # all but the lowest slice in secondary are all limestone, but x=0 is out of the mask. 4 cells x=2,y=2 are nan
    freq_df = mapper.data_frame_frequencies(freq_table)
    assert freq_df.iloc[0]["frequency"] == 5 # where secondary_lithology_3d_array is np.nan
    assert freq_df.iloc[1]["frequency"] == (3-1) * 4 * (5-4) - 1 # lowest 2ndary slice all clay, but x=0 is out of the mask, and one cell x=2,y=2 is nan
    assert freq_df.iloc[2]["frequency"] == ((3-1) * 4 * (5-1) - 4*1) # all but the lowest slice in secondary are all limestone, but x=0 is out of the mask. 4 cells x=2,y=2 are nan
    df = pd.DataFrame([('sand','clay'),('loam','')], columns=[PRIMARY_LITHO_COL, SECONDARY_LITHO_COL])
    litho_keys = mapper.create_full_litho_desc(df)
    assert litho_keys[0] == 'sand/clay'
    assert litho_keys[1] == 'loam/'
    # Secondary lithology outside of the known class is ignored, but we need a valid one for the primary lithology
    assert mapper.class_code('sand','martian rock') == mapper.class_code('sand','')
    assert np.isnan(mapper.class_code('martian rock','sand'))

    dims = (2,3,4)
    #dim_x,dim_y,dim_z = dims
    prim_litho = np.full(dims, np.nan)
    secd_litho = np.full(dims, np.nan)
    sand_code = 0
    clay_code = 1
    limestone_code = 2
    prim_litho[0,0,0] = sand_code
    secd_litho[0,0,0] = sand_code
    prim_litho[0,0,1] = sand_code
    secd_litho[0,0,1] = clay_code
    prim_litho[0,0,2] = clay_code
    secd_litho[0,0,2] = sand_code
    prim_litho[0,0,3] = clay_code
    mapped = mapper.map_classes(prim_litho, secd_litho)
    assert mapped[0,0,0] == fast
    assert mapped[0,0,1] == fast
    assert mapped[0,0,2] == medium
    assert mapped[0,0,3] == slow
    assert np.isnan(mapped[1,0,3])

def test_extract_single_lithology_class_3d():
    x = np.empty([3,4,5])
    class_value = 1.0
    x[:,:,:] = class_value
    x[1,:,:] = class_value + 2
    x[:,2,:] = class_value + 3
    x[:,:,3] = class_value + 4
    y = extract_single_lithology_class_3d(x, class_value)
    other_val = class_value - 1
    assert (y[1,:,:] == other_val).all()
    assert (y[:,2,:] == other_val).all()
    assert (y[:,:,3] == other_val).all()
    assert (y[0,:,:] == other_val).all()
    assert (y[:,0,:] == other_val).all()
    assert (y[:,:,0] == other_val).all()
    assert (y[-1,:,:] == other_val).all()
    assert (y[:,-1,:] == other_val).all()
    assert (y[:,:,-1] == other_val).all()


def test_grid_interpolation():
    n_depths = 10
    litho_logs = []
    easting_col = 'lat'
    northing_col = 'lon'
    depth_to_col = 'DePthTo'
    depth_from_col = 'DePthFrOm'
    litho_class_num_code_col = 'LithoNum'
    lat_max = lon_max = 4

    for i in range(lat_max):
        for j in range(lon_max):
            mock_obs = pd.DataFrame(
            {
                easting_col:np.full(n_depths, float(i)),
                northing_col:np.full(n_depths, float(j)),
                depth_from_col: - np.arange(0, n_depths, 1),
                depth_to_col: - np.arange(1, n_depths+1, 1),
                litho_class_num_code_col: np.full(n_depths, np.floor(i / 3)),
            })
            litho_logs.append(mock_obs)
    litho_logs = pd.concat(litho_logs)
    gi = GridInterpolation(easting_col=easting_col, northing_col=northing_col, depth_from_ahd_col=depth_from_col, depth_to_ahd_col=depth_to_col)

    n_neighbours=10
    ahd_min=-9
    ahd_max=-1

    x_min, x_max, y_min, y_max = (0, lat_max, 0, lon_max)
    grid_res = 0.25
    m = create_meshgrid_cartesian(x_min, x_max, y_min, y_max, grid_res)
    z_ahd_coords = np.arange(ahd_min,ahd_max,1)
    dim_x,dim_y = m[0].shape
    dim_z = len(z_ahd_coords)
    dims = (dim_x,dim_y,dim_z)

    lithology_3d_array=np.empty(dims)
    gi.interpolate_volume(lithology_3d_array, litho_logs, litho_class_num_code_col, z_ahd_coords, n_neighbours, m)
    assert lithology_3d_array[0,0,0] == 0.0

# test_numeric_code_mapping()
# test_get_lithology_observations_for_depth()
# test_make_training_set()
# test_get_knn_model()
# test_class_probability_estimates_depth()
# test_class_mapping()
# test_extract_single_lithology_class_3d()
# test_grid_interpolation()
