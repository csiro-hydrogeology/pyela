import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
from datetime import datetime
import rasterio
import geopandas as gpd

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.insert(0, pkg_dir)

from ela.textproc import *
from ela.spatial import *
from ela.classification import *
from ela.io import GeotiffExporter
from ela.utils import flip

from shapely.geometry import Point

def test_create_meshgrid():
    xx, yy = create_meshgrid_cartesian(x_min=0.0, x_max=1.1, y_min=1.0, y_max=1.51, grid_res = 0.5)
    assert xx.shape[0] == 3
    assert xx.shape[1] == 2
    assert yy.shape[0] == 3
    assert yy.shape[1] == 2

class MockSlicePredictor:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def f(self, x, y):
        return self.a * x + self.b * y + self.c

    def predict_one_sample(self, sample):
        x = sample[0]
        y = sample[1]
        return self.f(x, y)

    def predict(self, X):
        z = [self.predict_one_sample(x) for x in X]
        return np.array(z)

def test_interpolate_slice():
    m = create_meshgrid_cartesian(x_min=0.0, x_max=1.1, y_min=1.0, y_max=1.51, grid_res = 0.5)
    xx, yy = m
    a = 1.0
    b = 0.1
    c = 0.01
    p = MockSlicePredictor(a, b, c)
    def z_func(xi, yi):
        return p.f(xx[xi, yi], yy[xi, yi])


    predicted = interpolate_over_meshgrid(p, m)
    assert predicted.shape[0] == 3
    assert predicted.shape[1] == 2
    assert predicted[0,0] == z_func(0, 0)
    assert predicted[1,0] == z_func(1, 0)
    assert predicted[2,0] == z_func(2, 0)
    assert predicted[0,1] == z_func(0, 1)
    assert predicted[1,1] == z_func(1, 1)
    assert predicted[2,1] == z_func(2, 1)
    # work around scikit behavior:
    predicted = interpolate_over_meshgrid(None, m)
    assert predicted.shape[0] == 3
    assert predicted.shape[1] == 2
    assert np.isnan(predicted[1,1])

def test_height_coordinate_functor():
    z_index_for_ahd = z_index_for_ahd_functor(b=+100)
    assert z_index_for_ahd(-100) == 0
    assert z_index_for_ahd(-99) == 1
    assert z_index_for_ahd(0) == 100
    assert z_index_for_ahd(+50) == 150

def test_burn_volume():
    dims = (3,4,5)
    dim_x,dim_y,dim_z = dims
    x = np.arange(0.0, dim_x*dim_y*dim_z, 1.0)
    test_vol = np.reshape(x, dims)
    z_index_for_ahd = z_index_for_ahd_functor(b=+1) # z = 0 is datum height -1, z = 4 is datum height 3
    xx, yy = create_meshgrid_cartesian(x_min=0.0, x_max=0.51, y_min=0.0, y_max=0.76, grid_res = 0.25)
    dem = xx + yy
    assert dem[0,0] == 0.0
    assert dem[1,1] == 0.5
    assert dem[2,2] == 1.0
    burnt = test_vol.copy()
    burn_volume(burnt, dem, z_index_for_ahd, below=False, inclusive=False)
    assert not np.isnan(burnt[0,0,0])
    assert not np.isnan(burnt[0,0,1])
    assert np.isnan(burnt[0,0,2])

    assert not np.isnan(burnt[2,2,0])
    assert not np.isnan(burnt[2,2,1])
    assert not np.isnan(burnt[2,2,2])
    assert np.isnan(burnt[2,2,3])

    burnt = test_vol.copy()
    burn_volume(burnt, dem, z_index_for_ahd, below=False, inclusive=True)
    assert not np.isnan(burnt[0,0,0])
    assert np.isnan(burnt[0,0,1])
    assert np.isnan(burnt[0,0,2])

    assert not np.isnan(burnt[2,2,0])
    assert not np.isnan(burnt[2,2,1])
    assert np.isnan(burnt[2,2,2])
    assert np.isnan(burnt[2,2,3])

def test_slice_volume():
    dims = (3,4,5)
    dim_x,dim_y,dim_z = dims
    x = np.arange(0.0, dim_x*dim_y*dim_z, 1.0)
    test_vol = np.reshape(x, dims)
    dem = np.empty((3, 4))
    z_index_for_ahd = z_index_for_ahd_functor(b=+1) # z = 0 is datum height -1, z = 4 is datum height 3
    dem[0,0] = -2.0
    dem[0,1] = +5.0
    dem[0,2] = -1.0
    dem[0,3] = -1.0
    dem[1,0] = -1.0
    dem[1,1] = -1.0
    dem[1,2] = -1.0
    dem[1,3] = -1.0
    dem[2,0] = -1.0
    dem[2,1] = -1.0
    dem[2,2] = np.nan
    dem[2,3] = -1.0

    # TODO: I do not really like using volume_value_at. Make sure this is unit tested itself.
    def f(x, y):
        return volume_value_at(test_vol, dem, z_index_for_ahd, x, y)

    assert np.isnan(f(0,0))
    assert np.isnan(f(0,1))
    assert f(0,2) == test_vol[0,2,0]
    assert f(0,3) == test_vol[0,3,0]
    assert f(1,0) == test_vol[1,0,0]
    assert f(1,1) == test_vol[1,1,0]
    assert f(1,2) == test_vol[1,2,0]
    assert f(1,3) == test_vol[1,3,0]
    assert f(2,0) == test_vol[2,0,0]
    assert f(2,1) == test_vol[2,1,0]
    assert np.isnan(f(2,2))
    assert f(2,3) == test_vol[2,3,0]

    s = slice_volume(test_vol, dem, z_index_for_ahd)
    assert np.isnan(s[0,0])
    assert np.isnan(s[0,1])
    assert f(0,2) == s[0,2]
    assert f(0,3) == s[0,3]
    assert f(1,0) == s[1,0]
    assert f(1,1) == s[1,1]
    assert f(1,2) == s[1,2]
    assert f(1,3) == s[1,3]
    assert f(2,0) == s[2,0]
    assert f(2,1) == s[2,1]
    assert np.isnan(s[2,2])
    assert f(2,3) == s[2,3]

    sops = SliceOperation(dem, z_index_for_ahd)
    test_slices = sops.from_ahd_to_depth_below_ground_level(test_vol, from_depth=-1, to_depth=+1)
    s = test_slices
    assert s.shape[0] == dim_x
    assert s.shape[1] == dim_y
    assert s.shape[2] == 3
    index_ground_lvl = 1 # the top level is for depth=0 (dem), but it is at index 1 in the resulting volume s. one metre below ground level is what is at index 0 for the third dimension.
    assert np.isnan( s[0,0,index_ground_lvl])
    assert np.isnan( s[0,1,index_ground_lvl])
    assert f(0,2) == s[0,2,index_ground_lvl]
    assert f(0,3) == s[0,3,index_ground_lvl]
    assert f(1,0) == s[1,0,index_ground_lvl]
    assert f(1,1) == s[1,1,index_ground_lvl]
    assert f(1,2) == s[1,2,index_ground_lvl]
    assert f(1,3) == s[1,3,index_ground_lvl]
    assert f(2,0) == s[2,0,index_ground_lvl]
    assert f(2,1) == s[2,1,index_ground_lvl]
    assert np.isnan( s[2,2,index_ground_lvl])
    assert f(2,3) == s[2,3,index_ground_lvl]
    averaged_slices = sops.reduce_slices_at_depths(test_vol, from_depth=-1, to_depth=0, reduce_func=SliceOperation.arithmetic_average)
    s = averaged_slices
    assert np.isnan( s[0,0])
    assert np.isnan( s[0,1])
    # test_vol was constructed such that Z values increase by one at a given X/Y location, so the slicing/averaging result is like offsetting by 1/2:
    assert f(0,2) + 0.5 == s[0,2]
    assert f(0,3) + 0.5 == s[0,3]
    assert f(1,0) + 0.5 == s[1,0]
    assert f(1,1) + 0.5 == s[1,1]
    assert f(1,2) + 0.5 == s[1,2]
    assert f(1,3) + 0.5 == s[1,3]
    assert f(2,0) + 0.5 == s[2,0]
    assert f(2,1) + 0.5 == s[2,1]
    assert np.isnan( s[2,2])
    assert f(2,3) + 0.5 == s[2,3]

def get_test_bore_df():
    x_min = 383200
    y_max = 6422275
    return pd.DataFrame({
        EASTING_COL:np.array([x_min-.5,x_min+.5,x_min+1.1,x_min+1.1]),
        NORTHING_COL:np.array([y_max-0.1,y_max-0.1,y_max-0.9,y_max-1.1]),
        'fake_obs': np.array([.1, .2, .3, .4]),
        DEPTH_FROM_COL: np.array([1.11, 2.22, 3.33, 4.44]),
        DEPTH_TO_COL: np.array(  [2.22, 3.33, 4.44, 5.55])
        })


def create_test_slice(ni = 3, nj = 2, start=0.0, incr_1 = 1.0):
    return np.array([
        [(start + incr_1 * (i + ni * j)) for i in range(ni) ] for j in range(nj)
    ])

def get_slices_stack(n = 2, ni = 3, nj = 2, start=0.0, incr_1 = 1.0, incr_2 = 0.1):
    x = create_test_slice(ni, nj, start, incr_1)
    return [x + incr_2 * k for k in range(n)]

def create_test_raster(x_min = 383200, y_max = 6422275, grid_res = 1 , ni = 2, nj = 2, start=1.0, incr_1 = 1.0, output_file='c:/tmp/test_raster_drill.tif'):
    crs = rasterio.crs.CRS({'proj': 'utm', 'zone': 50, 'south': True, 'ellps': 'GRS80', 'units': 'm', 'no_defs': True})
    # Upper left hand corner is at (x_min, y_max), and in raster terms this is the origin.
    from rasterio.transform import from_origin
    transform = from_origin(x_min, y_max, grid_res, grid_res)
    ge = GeotiffExporter(crs, transform)
    # x = np.array([[1.0, 2.0],[3.0, 4.0]])
    x = create_test_slice(ni, nj, start, incr_1)
    ge.export_geotiff(x, output_file, None)

def test_raster_drill():
    # create_test_raster(x_min = 383200, y_max = 6422275, grid_res = 1 ,output_file='c:/tmp/test_raster_drill.tif'):
    x_min = 383200
    y_max = 6422275
    df = get_test_bore_df()
    dem = rasterio.open(os.path.join(pkg_dir, 'tests', 'data', 'test_raster_drill.tif'))
    cd = HeightDatumConverter(dem)
    heights = cd.raster_drill_df(df)
    assert np.isnan(heights[0])
    assert heights[1] == 1.0
    assert heights[2] == 2.0
    assert heights[3] == 4.0

def test_add_ahd():
    df = get_test_bore_df()
    dem = rasterio.open(os.path.join(pkg_dir, 'tests', 'data', 'test_raster_drill.tif'))
    cd = HeightDatumConverter(dem)
    df_ahd = cd.add_height(df)
    from_ahd = df_ahd[DEPTH_FROM_AHD_COL]
    to_ahd = df_ahd[DEPTH_TO_AHD_COL]
    assert np.isnan(from_ahd[0])
    assert from_ahd[1] == 1.0 - 2.22
    assert from_ahd[2] == 2.0 - 3.33
    assert from_ahd[3] == 4.0 - 4.44
    assert np.isnan(to_ahd[0])
    assert to_ahd[1] == 1.0 - 3.33
    assert to_ahd[2] == 2.0 - 4.44
    assert to_ahd[3] == 4.0 - 5.55


def test_average_slices():
    slices = get_slices_stack()
    avg = average_slices(slices)
    incr_2 = 0.1
    assert avg[0,0] == incr_2 / 2
    assert avg[0,1] == (incr_2 / 2) + 1.0
    assert avg[1,0] == (incr_2 / 2) + 3.0

# create_test_raster(x_min = 383200, y_max = 6422275, grid_res = 25 , ni = 10, nj = 10, start=0.0, incr_1 = 1.0, output_file=os.path.join(pkg_dir, 'tests', 'data', 'test_raster_25m.tif'))
def test_surface_array():
    dem = rasterio.open(os.path.join(pkg_dir, 'tests', 'data', 'test_raster_25m.tif'))
    grid_res = 100
    x_min = 383200 + 5 # first column
    y_max = 6422275 - 5 # meaning falls within first row/band from top
    y_min = y_max - (2 * grid_res) # 200 m offset over a 25m dem res, meaning falls within the 9th row/band from top
    x_max = x_min + (2 * grid_res) # 200 m offset over a 25m dem res, meaning falls within the 9th column from left
    surf_dem = surface_array(dem, x_min, y_min, x_max, y_max, grid_res)

    assert surf_dem.shape[0] == 2
    assert surf_dem.shape[1] == 2

    assert surf_dem[0,0] == 80 + 0.0
    assert surf_dem[0,1] == 80 + -4 * 10.0
    assert surf_dem[1,0] == 80 + 4.0
    assert surf_dem[1,1] == 80 + -4 * 10.0 + 4.0


def test_flip():
    m = np.zeros([2,3,4])
    m[1,2,3] = 3.14
    assert flip(m, 0)[0,2,3] == 3.14
    assert flip(m, 1)[1,0,3] == 3.14
    assert flip(m, 2)[1,2,0] == 3.14


def test_get_coords_from_gpd_shape():
    easting_values = np.array([.1, .2, .3, .3 ])
    northing_values = np.array([.12, .22, .32, .32 ])
    coords = get_unique_coordinates(easting_values, northing_values)
    assert coords.shape[0] == 3
    assert coords.shape[1] == 2
    ptsdf = pd.DataFrame({ 'Coordinates' : list(zip(coords[:,0], coords[:,1])) })
    ptsdf['Coordinates'] = ptsdf['Coordinates'].apply(Point)
    gdf = gpd.GeoDataFrame(ptsdf, geometry='Coordinates')
    gdf.crs = "+proj=utm +zone=56 +ellps=GRS80 +south +units=m +no_defs"
    geoloc = get_coords_from_gpd_shape(gdf, colname='Coordinates', out_colnames=['xx','yy'])
    x = geoloc.xx
    y = geoloc.yy
    assert len(x) == 3
    assert x[0] == .1
    assert x[1] == .2
    assert x[2] == .3
    assert y[0] == .12
    assert y[1] == .22
    assert y[2] == .32

def test_rounding_depths():
    n = 6
    df = pd.DataFrame({
        EASTING_COL:np.full(n, 1.1),
        NORTHING_COL:np.full(n, 2.2),
        'fake_obs': np.array([.1, .2, .3, .4, .5, .6]),
        DEPTH_FROM_COL: np.array([1.11, 1.16, 2.22, 3.33, 3.38, 4.44]),
        DEPTH_TO_COL: np.array(  [1.16, 2.22, 3.33, 3.38, 4.44, 5.55])
    })
    dr = DepthsRounding(DEPTH_FROM_COL, DEPTH_TO_COL)
    assert dr.assess_num_collapsed(df) == 2
    df_rd = dr.round_to_metre_depths(df, np.round, True)

    f = df_rd[DEPTH_FROM_COL].values
    t = df_rd[DEPTH_TO_COL].values

    assert f[0] == 1.0
    assert f[1] == 2.0
    assert f[2] == 3.0
    assert f[3] == 4.0
    assert t[0] == 2.0
    assert t[1] == 3.0
    assert t[2] == 4.0
    assert t[3] == 6.0


# test_add_ahd()
# test_flip()
# test_surface_array()
# test_average_slices()
# test_slice_volume()
# test_interpolate_slice()
# test_burn_volume()
# test_height_coordinate_functor()
# # test_make_training_set()
# test_raster_drill()
# test_get_coords_from_gpd_shape()
# test_rounding_depths()
