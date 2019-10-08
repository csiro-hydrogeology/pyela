import pyvista as pv
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import os
import sys

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

pkg_dir = '/home/per202/src/ela/pyela'

sys.path.insert(0, pkg_dir)

from ela.visual import discrete_classes_colormap
from ela.spatial import create_meshgrid_cartesian
from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

if ('ELA_DATA' in os.environ):
    data_path = os.environ['ELA_DATA']
elif sys.platform == 'win32':
    data_path = r'C:\data\Lithology'
else:
    username = os.environ['USER']
    data_path = os.path.join('/home', username, 'data', 'Lithology')

colnames = ['BoreID', 'Easting', 'Northing', 'Depth From (AHD)', 'Depth To (AHD)', 'Lithology_1_num']

h_0 = 630.0
n = 19

x_min = -200.0
x_max = +200.0
y_min = -200.0
y_max = +200.0
grid_res = 50.0
m = create_meshgrid_cartesian(x_min, x_max, y_min, y_max, grid_res)
xx, yy = m
def dem_height(x, y):
    return h_0 + 0.01 * x + 0.01 * y

dem_array = dem_height(xx, yy)
dem_data = {'bounds': (x_min, x_max, y_min, y_max), 'grid_res': grid_res, 'mesh_xy': m, 'dem_array': dem_array}

c = [(101, 0.0, 0.0, h_0 - i * 2, h_0 - 2 - i * 2, float(i)) for i in range(n)]

h_1 = dem_height(100, 100)
c = c + [(202, 100.0, 100.0, h_1 - i * 2, h_1 - 2 - i * 2, float(i)) for i in range(n//2)]

h_2 = dem_height(-100, -100)
# Craft test to check the interpolation (non contiguous colors)
c = c + [
    (303, -100.0, -100.0, h_2 - 0 * 2, h_2 - 2 - 0 * 2, 00.0), 
    (303, -100.0, -100.0, h_2 - 1 * 2, h_2 - 2 - 1 * 2, 18.0), 
    (303, -100.0, -100.0, h_2 - 2 * 2, h_2 - 2 - 2 * 2, 9.0), 
    ]

# Craft test to check the interpolation with missing data
h_3 = dem_height(-100, -100)
c = c + [
    (404, -50.0, -100.0, h_3 - 0 * 2, h_3 - 2 - 0 * 2, 00.0), 
    (404, -50.0, -100.0, h_3 - 1 * 2, h_3 - 2 - 1 * 2, 18.0), 
    (404, -50.0, -100.0, h_3 - 2 * 2, h_3 - 2 - 2 * 2, np.nan), 
    (404, -50.0, -100.0, h_3 - 3 * 2, h_3 - 2 - 3 * 2, 00.0), 
    (404, -50.0, -100.0, h_3 - 4 * 2, h_3 - 2 - 4 * 2, 18.0), 
    (404, -50.0, -100.0, h_3 - 5 * 2, h_3 - 2 - 5 * 2, 09.0), 
    ]


h_4 = dem_height(-50, -50)

# Craft test to check the benavior with unordered data
ordering = [2,0,1,3]
litho_class = {0: 0.0, 1: np.nan, 2: 2.0, 3: 3.0}
c = c + [(505, -50.0, -50.0, h_4 - i * 2, h_4 - 2 - i * 2, litho_class[i]) for i in ordering]

drill_data= pd.DataFrame(c, columns=colnames)

# drill_data_path = os.path.join(data_path, 'Canberra','out','classified_logs.pkl')
# dem_data_path = os.path.join(data_path, 'Canberra','out','dem_array_data.pkl')
# drill_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\cbr\classified_logs.pkl"
# dem_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\cbr\dem_array_data.pkl"

dp = VisualizeDataProcess()
# drill_data = dp.drill_file_read(drill_data_path)
# dem_data = dp.dem_file_read(dem_data_path)

lines_dict = dp.drill_data_process(drill_data, 25)
# temp = dp.drill_file_read(drill_data_path)
# pd.set_option('display.max_columns', None)
# print(temp)

grid = dp.dem_data_process(dem_data, 25)

annotations =dict([(float(i), str(i)) for i in range(n)] )

sargs = dict(
    n_labels=len(annotations),
    bold=False,
    interactive=False,
    label_font_size=8,
    fmt="%.1f",
    font_family="arial",
    vertical=True,
    position_x=1,
    position_y=0.45,
    height=0.5,
)
plotter = pv.Plotter()
for well in lines_dict.keys():
    plotter.add_mesh(lines_dict.get(well),
                     scalars=dp.scalar_prop_name,
                     scalar_bar_args=sargs,
                     annotations=annotations,
                     show_edges=False,
                     edge_color="white",
                     n_colors=len(annotations),
                     nan_color="black",
                     clim=[0, len(annotations)-1],
                     opacity=1,
                     )

plotter.add_mesh(grid, opacity=0.9)
plotter.show_bounds(grid, show_xaxis=True, show_yaxis=True, show_zaxis=False)
plotter.show_axes()
plotter.show()
