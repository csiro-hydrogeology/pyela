import time
import pyvista as pv
from ela.visual import discrete_classes_colormap
import numpy as np
from matplotlib.colors import ListedColormap
from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

drill_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\classified_logs.pkl"
dem_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\dem_array_data.pkl"

dp = VisualizeDataProcess()
lines_dict = dp.drill_data_process(drill_data_path, 25)
grid = dp.dem_data_process(dem_data_path, 25)
layer = dp.lithology_layer_process(drill_data_path, dem_data_path, 25, 6, 10)

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
                     scalars="GR",
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
