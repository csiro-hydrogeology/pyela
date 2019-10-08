import time
import pyvista as pv
import os
import sys
import pandas as pd
import numpy as np

pkg_dir = os.path.join(os.path.dirname(__file__),'..')
sys.path.insert(0, pkg_dir)

from ela.visual import discrete_classes_colormap
import numpy as np
from matplotlib.colors import ListedColormap
from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

# start = time.clock()
'''
A sample of 3D image based on pyvista

'''

if ('ELA_DATA' in os.environ):
    data_path = os.environ['ELA_DATA']
elif sys.platform == 'win32':
    data_path = r'C:\data\Lithology'
else:
    username = os.environ['USER']
    data_path = os.path.join('/home', username, 'data', 'Lithology')

drill_data_path = os.path.join(data_path, 'Bungendore','classified_logs.pkl')
df = pd.read_pickle(drill_data_path)

# drill_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\classified_logs.pkl"
bungendore_datadir = os.path.join(data_path, 'Bungendore')
dem_data_path = os.path.join(bungendore_datadir, 'dem_array_data.pkl')

dp = VisualizeDataProcess()
drill_data = dp.drill_file_read(drill_data_path)
dem_data = dp.dem_file_read(dem_data_path)
lines_dict = dp.drill_data_process(drill_data, 25, min_tube_radius = 70)
grid = dp.dem_data_process(dem_data, 25)
layer = dp.lithology_layer_process(drill_data, dem_data, 'Bungendore', 25, 5, 10)
print(type(layer))

# ['clay','sand','gravel','granite','shale','silt','topsoil','loam','soil','slate','sandstone']
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

plotter = pv.Plotter()

lithologies = ['clay', 'sand', 'gravel', 'granite', 'shale', 'silt', 'topsoil', 'loam', 'soil', 'slate', 'sandstone']
lithology_color_names = ['olive', 'yellow', 'lightgrey', 'dimgray', 'teal', 'cornsilk', 'saddlebrown', 'rosybrown',
                         'chocolate', 'lightslategrey', 'gold']
lithology_cmap = discrete_classes_colormap(lithology_color_names)
mapping = np.linspace(0, 10, 11)
newcolors = np.empty((11, 4))
for i in range(11):
    newcolors[i] = lithology_cmap.get(i)
    newcolors[i] = newcolors[i] / 256
my_colourmap = ListedColormap(newcolors)

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
    # plotter.add_mesh_clip_plane(lines_dict[well])

# plotter.add_mesh_clip_plane(plotter)
# plotter.add_scalar_bar("Lithology", 11, interactive=True)
plotter.add_mesh(layer, scalars="Lithology", n_colors=11, clim=[0, 10], show_scalar_bar=False)
# plotter.add_mesh_clip_box(layer)
plotter.add_mesh(grid, opacity=0.9)
plotter.show_bounds(grid, show_xaxis=True, show_yaxis=True, show_zaxis=False)
plotter.show_axes()
# end = time.clock()
# print(end - start)
plotter.show()
# ['clay','sand','gravel','granite','shale','silt','topsoil','loam','soil','slate','sandstone']
