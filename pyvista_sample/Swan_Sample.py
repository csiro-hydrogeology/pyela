import pyvista as pv
import pandas as pd
from ela.visual import discrete_classes_colormap
import numpy as np
from matplotlib.colors import ListedColormap
from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

# start = time.clock()
'''
A sample of 3D image based on pyvista

'''
drill_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\swan_coastal\classified_logs.pkl"
dem_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\swan_coastal\dem_array_data.pkl"

dp = VisualizeDataProcess()
drill_data = dp.drill_file_read(drill_data_path)
dem_data = dp.dem_file_read(dem_data_path)
lines_dict = dp.drill_data_process(drill_data, 25)
# temp = dp.drill_file_read(drill_data_path)
# pd.set_option('display.max_columns', None)
# print(temp)

grid = dp.dem_data_process(dem_data, 25)

layer = dp.lithology_layer_process(drill_data, dem_data, 'swan', 25, 7, 10)

annotations = {
    0.00: "Sand",
    1.00: "Clay",
    2.00: "Quartz",
    3.00: "Shale",
    4.00: "Sandstone",
    5.00: "Coal",
    6.00: "Pebbles",
    7.00: "silt",
    8.00: "pyrite",
    9.00: "Grit",
    10.00: "limestone",
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
plotter = pv.Plotter()
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

plotter.add_mesh(layer, scalars="Lithology", n_colors=11, clim=[0, 10], show_scalar_bar=False)
plotter.add_mesh(grid, opacity=0.9)
plotter.show_bounds(grid, show_xaxis=True, show_yaxis=True, show_zaxis=False)
plotter.show_axes()
plotter.show()
