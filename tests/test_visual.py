import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
from datetime import datetime

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.append(pkg_dir)

from ela.visual import *

def test_cartopy_cms():
    color_names = ['blue','red','green']
    cms = cartopy_color_settings(color_names)
    assert set(cms.keys()) == set(['cmap','bounds','norm'])
    cmap = cms['cmap']
    bounds  = cms['bounds']
    assert len(bounds) == 4
    assert bounds[0] == 0.0
    assert 0.999 < bounds[1]
    assert 1.999 < bounds[2]
    assert 2.999 < bounds[3]
    assert len(cmap.colors) == 3
    assert cmap.colors[0] == 'blue'
    assert cmap.colors[1] == 'red'
    assert cmap.colors[2] == 'green'


# def test_visual_info():
#     class_names = ['high_k','low_k']
#     color_names = ['blue','red']
#     blah = LithologiesClassesVisual(class_names, color_names, missing_value_color_name)

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

