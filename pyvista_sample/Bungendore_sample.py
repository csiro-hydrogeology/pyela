import pyvista as pv
from pyvista_sample.VisualizeDataProcess import VisualizeDataProcess

'''
A sample of 3D image based on pyvista

'''
drill_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\classified_logs.pkl"
dem_data_path = r"C:\Users\Dennis.H\Desktop\CSIRO_data\Bungendore\dem_array_data.pkl"

dp = VisualizeDataProcess()
lines_dict = dp.drill_data_process(drill_data_path, 25)
grid = dp.dem_data_process(dem_data_path, 25)

plotter = pv.Plotter()
for well in lines_dict:
    plotter.add_mesh(lines_dict[well])
plotter.add_mesh(grid, opacity=0.7)

# plotter.add_mesh(pierregrid, opacity=0.5, color="green")

plotter.show()
