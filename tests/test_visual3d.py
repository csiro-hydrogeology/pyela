# See issue https://github.com/csiro-hydrogeology/pyela/issues/12

import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
from datetime import datetime
from pyvista_sample.VisualizeDataProcess import *

pkg_dir = os.path.join(os.path.dirname(__file__), '..')

sys.path.append(pkg_dir)

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


# origin_columns = ['','Depth From (AHD)', 'Depth To (AHD)']
def test_add_missing_height_data():
    df = vp.add_scaled_height_column(df_1, 20)
    well_dict = vp.build_well_dict(df)
    test1 = len(well_dict.get('1')["scaled_from_height"].values)
    test2 = len(well_dict.get('2')["scaled_from_height"].values)
    # print(test1)
    temp = vp.add_missing_height_data(well_dict)
    assert len(temp.get('1')["scaled_from_height"].values) == test1 + 1
    assert len(temp.get('2')["scaled_from_height"].values) == test2 + 1


def test_build_points_dict():
    df = vp.add_scaled_height_column(df_1, 20)
    well_dict = vp.build_well_dict(df)
    well_dict = vp.add_missing_height_data(well_dict)
    point_dict = vp.build_points_dict(well_dict)
    test1 = np.array([10, 10, 20])
    test2 = np.array([20, 20, 800])
    # print(point_dict.get('1')[0])
    # print(point_dict.get('2')[1])
    assert point_dict.get('1')[0][0] == test1[0]
    assert point_dict.get('2')[1][2] == test2[2]


def test_add_lithology_based_scalar():
    df = vp.add_scaled_height_column(df_1, 20)
    well_dict = vp.build_well_dict(df)
    well_dict = vp.add_missing_height_data(well_dict)
    point_dict = vp.build_points_dict(well_dict)
    # print(well_dict.get('1'))
    # print(point_dict)
    line_dict = vp.Point_to_lines_dict(point_dict)
    lines_dict = vp.add_lithology_based_scalar(well_dict, line_dict)
    # print(lines_dict['2']["GR"])
    # print(lines_dict['1']["GR"])
    assert 1, 2 in lines_dict['1']["GR"]
    assert 3, 5 in lines_dict['2']["GR"]


