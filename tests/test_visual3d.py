# See issue https://github.com/csiro-hydrogeology/pyela/issues/12

import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, pkg_dir)

from pyvista_sample.VisualizeDataProcess import *

# from ela.visual3d import *

# def test_mlab_ui():
#     class_names = [
#         'class_1',
#         'class_2',
#         'class_3',
#         'class_4'
#         ]
#     color_names = ['red','orange','yellow','blue']
#     vis = LithologiesClassesVisual3d(class_names, color_names, missing_value_color_name='black')
#     assert vis.nb_labels() == len(class_names)
#     assert vis.nb_labels_with_missing() == len(class_names) + 1
#     volume = np.empty([2,3,4], dtype='float64')
#     volume[:] = 0.0
#     volume[:,:,1] = 1.0
#     volume[:,:,2] = 2.0
#     volume[:,:,3] = 3.0
#     vis.render_classes_planar(volume,'blah title')

origin_columns = ['BoreID', 'Easting', 'Northing', 'Depth From (AHD)', 'Depth To (AHD)', 'Lithology_1_num']
index = [1, 2, 3, 4]
origin_data = np.array([[1, 10, 10, 20, 10, 1], [1, 10, 10, 10, 0, 2], [2, 20, 20, 20, 10, 3], [2, 20, 20, 40, 20, 5]])
df_1 = pd.DataFrame(origin_data, index=index, columns=origin_columns)
# print(df_1)
vp = VisualizeDataProcess()


def test_add_scaled_column():
    test1 = [400, 200]
    test2 = [200, 0]
    temp = vp.add_scaled_height_column(df_1, 20)
    # print(temp["scaled_from_height"].values)
    assert temp["scaled_from_height"].values[0] == test1[0]
    assert temp["scaled_from_height"].values[1] == test1[1]
    assert temp["scaled_to_height"].values[0] == test2[0]
    assert temp["scaled_to_height"].values[1] == test2[1]


def test_build_well_dict():
    df = vp.add_scaled_height_column(df_1, 20)
    temp = vp.build_well_dict(df)
    test1 = temp.get('1')
    test2 = temp.get('2')
    assert test1["scaled_from_height"].values[0] == 400
    assert test2['Lithology_1_num'].values[1] == 5

def test_build_points_dict():
    df = vp.add_scaled_height_column(df_1, 20)
    well_dict = vp.build_well_dict(df)
    point_dict = vp.build_points_dict(well_dict)
    test1 = np.array([10, 10, 20])
    test2 = np.array([20, 20, 800])
    # print(point_dict.get('1')[0])
    # print(point_dict.get('2')[1])
    assert point_dict.get('1')[0][0] == test1[0]
    assert point_dict.get('2')[1][2] == test2[2]

def test_tubular_info_generation():
    '''Entries are such that we have entries for the top and bottom of the
    lithology log entries, so that the visual representation is more faithful to reality'''
    ordering = [2,0,3,1]
    litho_class = {0: 0.0, 1: np.nan, 2: 2.0, 3: 3.0}
    c = [(1234, 0.0, 0.0, -i , -i-1, litho_class[i]) for i in ordering]
    df_syn = pd.DataFrame(c, columns=origin_columns)
    df = vp.add_scaled_height_column(df_syn, 20)
    well_dict = vp.build_well_dict(df)
    point_dict = vp.build_points_dict(well_dict)
    assert len(point_dict['1234']) == 2 * len(df_syn)

def test_add_lithology_based_scalar():
    df = vp.add_scaled_height_column(df_1, 20)
    well_dict = vp.build_well_dict(df)
    point_dict = vp.build_points_dict(well_dict)
    # print(well_dict.get('1'))
    # print(point_dict)
    line_dict = vp.point_to_lines_dict(point_dict)
    lines_dict = vp.add_lithology_based_scalar(well_dict, line_dict)
    # print(lines_dict['2']["GR"])
    # print(lines_dict['1']["GR"])
    assert 1, 2 in lines_dict['1']["GR"]
    assert 3, 5 in lines_dict['2']["GR"]

# def test_todo():
#     # https://github.com/csiro-hydrogeology/pyela/issues/19
#     # layer = vp.lithology_layer_process(drill_data_path, dem_data_path, 25, 5, 10)
#     raise NotImplementedError()
