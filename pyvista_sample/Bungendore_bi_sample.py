import time
import pyvista as pv
import os
import sys
import numpy as np
from matplotlib.colors import ListedColormap

pkg_dir = os.path.join(os.path.dirname(__file__),'..')
sys.path.insert(0, pkg_dir)
from ela.visual import discrete_classes_colormap
from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

drill_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\classified_logs.pkl"
dem_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\dem_array_data.pkl"

if ('ELA_DATA' in os.environ):
    data_path = os.environ['ELA_DATA']
elif sys.platform == 'win32':
    data_path = r'C:\data\Lithology'
else:
    username = os.environ['USER']
    data_path = os.path.join('/home', username, 'data', 'Lithology')

drill_data_path = os.path.join(data_path, 'Bungendore','classified_logs.pkl')
bungendore_datadir = os.path.join(data_path, 'Bungendore')
dem_data_path = os.path.join(bungendore_datadir, 'dem_array_data.pkl')

dp = VisualizeDataProcess()
drill_data = dp.drill_file_read(drill_data_path)
dem_data = dp.dem_file_read(dem_data_path)
lines_dict = dp.drill_data_process(drill_data, 25, min_tube_radius = 50)
grid = dp.dem_data_process(dem_data, 25)
layer = dp.lithology_layer_process(drill_data, dem_data, 'Bungendore', 25, 5, 10)

annotations = {
    0.00: "Clay",
    1.00: "Sand",
    2.00: "gravel",
    3.00: "Granite",
    4.00: "Shale",
    5.00: "Silt",
    6.00: "Topsoil",
    7.00: "Loam",
    8.00: "Soil",
    9.00: "Slate",
    10.00: "Sandstone",
}

sargs = dict(
    n_labels=11,
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

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
for well in lines_dict.keys():
    plotter.add_mesh(lines_dict.get(well),
                     scalars=dp.scalar_prop_name,
                     scalar_bar_args=sargs,
                     annotations=annotations,
                     show_edges=False,
                     edge_color="white",
                     n_colors=11,
                     nan_color="black",
                     clim=[0, 10],
                     opacity=1,
                     )
plotter.add_mesh(grid, opacity=0.9)
plotter.show_bounds(grid, show_xaxis=True, show_yaxis=True, show_zaxis=False)
plotter.show_axes()

plotter.subplot(0, 1)

plotter.add_mesh(layer, scalars="Lithology", n_colors=11, clim=[0, 10], scalar_bar_args=sargs, annotations=annotations)
plotter.add_mesh(grid, opacity=0.9)
plotter.show_bounds(grid, show_xaxis=True, show_yaxis=True, show_zaxis=False)
plotter.show_axes()

plotter.show()
