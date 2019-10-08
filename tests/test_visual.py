import os
import numpy as np
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__), '..')

sys.path.insert(0, pkg_dir)

from ela.visual import *


def test_cartopy_cms():
    color_names = ['blue', 'red', 'green']
    cms = cartopy_color_settings(color_names)
    assert set(cms.keys()) == set(['cmap', 'bounds', 'norm'])
    cmap = cms['cmap']
    bounds = cms['bounds']
    assert len(bounds) == 4
    assert bounds[0] == 0.0
    assert 0.999 < bounds[1]
    assert 1.999 < bounds[2]
    assert 2.999 < bounds[3]
    assert len(cmap.colors) == 3
    assert cmap.colors[0] == 'blue'
    assert cmap.colors[1] == 'red'
    assert cmap.colors[2] == 'green'


def test_to_carto():
    xy = np.array([[1, 2, 3], [4, 5, 6]], dtype='uint8')
    # in x-y coordinates where xy indexing is [x,y]
    # ^
    # | 3  6
    # | 2  5
    # | 1  4
    # . ------->
    d = to_carto(xy)
    assert len(d.shape) == 2
    assert d.shape[0] == 3
    assert d.shape[1] == 2
    # in cartopy screen coordinates where indexing is row,column
    # . ------->
    # | 3  6
    # | 2  5
    # | 1  4
    # \/
    assert d[0, 0] == 3
    assert d[0, 1] == 6
    assert d[1, 0] == 2
    assert d[1, 1] == 5
    assert d[2, 0] == 1
    assert d[2, 1] == 4


def test_get_color_component():
    c = np.empty([2, 3, 4])
    c[:, :, 0] = 0
    c[:, :, 1] = 100
    c[:, :, 2] = 200
    c[:, :, 3] = 255
    for x in range(2):
        for y in range(3):
            assert get_color_component(c, 0)[x, y] == 0
            assert get_color_component(c, 1)[x, y] == 100
            assert get_color_component(c, 2)[x, y] == 200
            assert get_color_component(c, 3)[x, y] == 255


def test_discrete_classes_colormap():
    cnames = ['blue', 'red', 'green']
    d = discrete_classes_colormap(cnames)
    assert len(d) == 3
    assert len(d[0]) == 4
    assert d[0][0] == 0  # r
    assert d[0][1] == 0  # g
    assert d[0][2] == 255  # b
    assert d[0][3] == 255  # a


def test_color_interpolation():
    blue = to_rgba_255('blue')
    red = to_rgba_255('red')
    cmap = {
        0: (0, 4, 128, 255),
        1: blue,
        2: red,
        3: (0, 10, 128, 100),
        4: (0, 20, 128, 100),
        5: (0, 30, 128, 100)
    }
    c = interpolate_rgba(cmap, 1, .4)
    assert c[0] == 102  # 255 * 0.4
    assert c[1] == 0
    assert c[2] == 153  # 255 * 0.6
    assert c[3] == 255


def test_to_color_image():
    cnames = ['blue', 'red', 'green']
    d = discrete_classes_colormap(cnames)
    x = np.array([[0, 2, 1], [2, 0, np.nan]], dtype='float64')
    image = to_color_image(x, d)
    assert len(image.shape) == 3
    def f(actual, expected):
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert actual[2] == expected[2]
        assert actual[3] == expected[3]
    f(image[0,0], [0,0,255,255]) # 0 is blue
    f(image[0,1], [0,128,0,255]) # 2 is green
    f(image[0,2], [255,0,0,255]) # 1 is red
    f(image[1,0], [0,128,0,255])
    f(image[1,1], [0,0,255,255])
    f(image[1,2], [255,255,255,255]) # nan is white

    f(image[0, 0], [0, 0, 255, 255])  # 0 is blue
    f(image[0, 1], [0, 128, 0, 255])  # 2 is green
    f(image[0, 2], [255, 0, 0, 255])  # 1 is red
    f(image[1, 0], [0, 128, 0, 255])
    f(image[1, 1], [0, 0, 255, 255])
    f(image[1, 2], [255, 255, 255, 255])  # nan is white


def test_rgba_conversions():
    rgbx = to_rgb('green')
    rgb = to_rgb_255('green')
    rgba = to_rgba_255('green')
    assert len(rgb) == 3
    assert len(rgba) == 4
    assert rgbx[0] == 0
    assert rgbx[1] == 128.0 / 255
    assert rgbx[2] == 0
    assert rgb[0] == 0
    assert rgb[1] == 128
    assert rgb[2] == 0
    assert rgba[0] == 0
    assert rgba[1] == 128
    assert rgba[2] == 0
    assert rgba[3] == 255


# solely for UT coverage, basically is it running...
def test_legend_fig():
    k_legend_display_info = [
        (None, 'class_one', 'gray'),
        (None, 'class_two', 'blue'),
        (None, 'class_three', 'red')
    ]
    fig = legend_fig(k_legend_display_info)
    assert fig is not None


def test_visual_info():
    class_names = ['high_k', 'low_k']
    color_names = ['blue', 'red']
    missing_value_color_name = 'black'
    lcv = LithologiesClassesVisual(class_names, color_names, missing_value_color_name)
    assert lcv.nb_labels() == 2
    assert lcv.nb_labels_with_missing() == 3


test_visual_info()
