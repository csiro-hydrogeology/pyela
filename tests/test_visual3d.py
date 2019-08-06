# See issue https://github.com/csiro-hydrogeology/pyela/issues/12

# import os
# import pandas as pd
# import numpy as np
# import datetime as dt
# import sys
# from datetime import datetime

# pkg_dir = os.path.join(os.path.dirname(__file__),'..')

# sys.path.append(pkg_dir)

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

