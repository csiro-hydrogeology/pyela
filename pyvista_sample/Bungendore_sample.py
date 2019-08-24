import pyvista as pv
import os
import sys
import pandas as pd
import numpy as np

pkg_dir = os.path.join(os.path.dirname(__file__),'..')

sys.path.append(pkg_dir)

from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

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
lines_dict = dp.drill_data_process(drill_data_path, 25)
grid = dp.dem_data_process(dem_data_path, 25)

plotter = pv.Plotter()
for well in lines_dict:
    plotter.add_mesh(lines_dict[well])
plotter.add_mesh(grid, opacity=0.7)

# plotter.add_mesh(pierregrid, opacity=0.5, color="green")

plotter.show()
