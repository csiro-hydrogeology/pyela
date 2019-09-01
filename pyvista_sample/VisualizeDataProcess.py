import pickle
import PVGeo
import pyvista as pv
import pandas as pd
from ela.classification import GridInterpolation

from ela.spatial import create_meshgrid_cartesian
from ela.visual import *

'''
@author: Guanjie Huang
@date: Aug 16th,2019

This class is used to process data before generating the 3D images

'''


class VisualizeDataProcess:
    def __init__(self):
        # self.height_Adjustment_factor=height_Adjustment_factor
        self.scaled_from_height_colname = 'scaled_from_height'
        self.scaled_to_height_colname = 'scaled_to_height'
        self.dem_x_min = 0
        self.dem_x_max = 0
        self.dem_y_min = 0
        self.dem_y_max = 0
        self.ahd_max = 0
        self.ahd_min = 0

    def drill_data_process(self, file_path, height_adjustment_factor=20):
        """The whole data process from drill data to PolyData dictionary

            Args:
                file_path (str): drill data file path
                height_adjustment_factor (int): Height scaling factor, default 20 .

            Returns:
                lines_dict(dict): PolyData dictionary.
        """
        data = self.drill_file_read(file_path)
        data = self.add_scaled_height_column(data, height_adjustment_factor)
        well_dict = self.build_well_dict(data)
        well_dict = self.add_missing_height_data(well_dict)
        point_dict = self.build_points_dict(well_dict)
        lines_dict = self.Point_to_lines_dict(point_dict)
        lines_dict = self.add_lithology_based_scalar(well_dict, lines_dict)
        return lines_dict

    def dem_data_process(self, file_path, height_adjustment_factor):
        """The whole data process from dem data to pv.StructuredGrid

            Args:
                file_path (str): dem data file path
                height_adjustment_factor (int): Height scaling factor, default 20 .

            Returns:
                Grid(pyvista.core.pointset.StructuredGrid)
        """
        dem_array_data = self.dem_file_read(file_path)
        xx, yy = dem_array_data['mesh_xy']
        dem_array = dem_array_data['dem_array']
        grid = pv.StructuredGrid(xx, yy, dem_array * height_adjustment_factor)
        return grid

    def lithology_layer_process(self, drill_file_path, dem_file_path, height_adjustment_factor=20, layer_from=0,
                                layer_to=0):
        """add points lithology type, expands lines to tube based on lithology number
            Args:
                drill_file_path(str): drill file path
                dem_file_path(str):dem file path
                height_adjustment_factor(int): height scala factor
                layer_from (float): set the begin number of layers
                layer_to (float): set the end number of layers
            Returns:
                layer_mesh(pyvista.core.pointset.UnstructuredGrid): layer mesh for display use
        """

        drill_data = self.drill_file_read(drill_file_path)
        dem_array_data = self.dem_file_read(dem_file_path)
        lithology_3d_array = self.build_layer_data(drill_data, dem_array_data)
        lithology_3d_array = self.clean_over_bound_data(lithology_3d_array, dem_array_data)
        # lithology_3d_array= self.vag_clean(lithology_3d_array,dem_array_data)
        layer_mesh = self.build_layer_mesh(lithology_3d_array, height_adjustment_factor, layer_from, layer_to)
        return layer_mesh

    def drill_file_read(self, file_path):
        """Read drill data file
            Args:
                file_path (str): drill data file path

            Returns:
                df(pandas.core.frame.DataFrame)
        """
        df = pd.read_pickle(file_path)
        self.ahd_max = df[DEPTH_FROM_AHD_COL].max()
        self.ahd_min = df[DEPTH_TO_AHD_COL].min()
        return df

    def dem_file_read(self, file_path):
        """Read dem data file
            Args:
                file_path (str): drill data file path

            Returns:
                dem_array_date(pandas.core.frame.DataFrame)
        """
        with open(file_path, 'rb') as handle:
            dem_array_data = pickle.load(handle)
        self.dem_x_min, self.dem_x_max, self.dem_y_min, self.dem_y_max = dem_array_data['bounds']
        self.grid_res = dem_array_data['grid_res']
        return dem_array_data

    def add_scaled_height_column(self, data, height_adjustment_factor):
        """Add scaled height columns to data frame
            Args:
                data (pandas.core.frame.DataFrame):original data
                height_adjustment_factor (int): Height scaling factor.
            Returns:
                data(pandas.core.frame.DataFrame): modified data
        """
        # scaled_from_height_colname = 'scaled_from_height'
        data[self.scaled_from_height_colname] = data[DEPTH_FROM_AHD_COL] * height_adjustment_factor
        # scaled_to_height_colname = 'scaled_to_height'
        data[self.scaled_to_height_colname] = data[DEPTH_TO_AHD_COL] * height_adjustment_factor
        return data

    def build_well_dict(self, data):
        """build dictionary according to BoreID
            Args:
                data (pandas.core.frame.DataFrame):original data
            Returns:
                well_dict(dict()): wells dictionary
        """
        data['name'] = data.BoreID.values.astype(str)
        wells = data.name.unique()
        well_dict = {}
        for well in wells:
            well_dict["{0}".format(well)] = data[data.name == well]
        return well_dict

    def add_missing_height_data(self, well_dict):
        """Add the smallest height_to data to height_from data (len(well_dict[i])+1)
            Args:
                well_dict(dict()): original dictionary
            Returns:
                well_dict(dict()): modified dictionary
        """
        for well in well_dict.keys():
            origin_well_df = well_dict.get(well)
            after_well_df = origin_well_df.copy()
            add_index = origin_well_df['scaled_to_height'].idxmin()
            line = origin_well_df.loc[add_index].copy()
            line.scaled_from_height = line.scaled_to_height
            line = line.to_frame()
            temp = []
            for value in line.values:
                if value[0]:
                    temp.append(value[0])
                else:
                    temp.append(0)
            after_well_df.loc["new"] = temp
            well_dict[well] = after_well_df
        return well_dict

    def build_points_dict(self, well_dict):
        """build points dictionary from wells dictionary
            Args:
                well_dict(dict()): wells dictionary
            Returns:
                points_dict(dict()): zip points axis for points
        """
        points_dict = {}
        for points in well_dict:
            points_dict["{0}".format(points)] = np.array(
                list(
                    zip(
                        well_dict[points].Easting,
                        well_dict[points].Northing,
                        # well_dict[points][self.scaled_from_height_colname],
                        well_dict[points].scaled_from_height
                    )
                )
            )
        return points_dict

    def Point_to_lines_dict(self, points_dict):
        """build lines dictionary from points dictionary
            Args:
                points_dict(dict()): points dictionary
            Returns:
                lines_dict(dict()): build lines between same well points
        """
        lines_dict = {}
        for lines in points_dict:
            poly = PVGeo.points_to_poly_data(points_dict[lines])
            lines_dict["{0}".format(
                lines)] = PVGeo.filters.AddCellConnToPoints(nearest_nbr=True).apply(poly)
            # notice that the building of the lines need to follow the nearest neighbourhood search
        return lines_dict

    def add_lithology_based_scalar(self, well_dict, lines_dict):
        """add points lithology type, expands lines to tube based on lithology number
            Args:
                well_dict(dict()): wells dictionary
                lines_dict(dict()):lines dictionary
            Returns:
                lines_dict(dict()): with new attribute "GR" which represent lithology number, and expanded to tube.
        """
        lines_dict_tmp = {}
        litho_class_col = 'Lithology_1_num'
        site_ref_colname = 'name'
        for path in lines_dict:
            vals = well_dict[path].Lithology_1_num.values
            lines_dict[path]["GR"] = vals
            lines_dict[path].tube(radius=10, scalars="GR", inplace=True)
            if len(vals) > 0:
                lines_dict_tmp[path] = lines_dict[path]
        lines_dict = lines_dict_tmp
        return lines_dict

    def build_layer_data(self, drill_data, dem_array_data):
        n_neighbours = 10
        """get the layer data from the function contains in ela
            Args:
                drill_data (pandas.core.frame.DataFrame): drill data
                dem_array_data (pandas.core.frame.DataFrame): dem data
        """
        xg, yg = dem_array_data['mesh_xy']
        m = create_meshgrid_cartesian(self.dem_x_min, self.dem_x_max, self.dem_y_min, self.dem_y_max, self.grid_res)
        z_coords = np.arange(self.ahd_min, self.ahd_max, 1)
        dim_x, dim_y = xg.shape
        dim_z = len(z_coords)
        dims = (dim_x, dim_y, dim_z)
        lithology_3d_array = np.empty(dims)
        gi = GridInterpolation(easting_col='x', northing_col='y')
        gi.interpolate_volume(lithology_3d_array, drill_data, PRIMARY_LITHO_NUM_COL, z_coords, n_neighbours, m)
        return lithology_3d_array

    def clean_over_bound_data(self, lithology_3d_array, dem_array_data):
        """accurate process data that exceeds limits
        （we suppose that the lithology would not higher than the ground surface),
        accurate but slower
            Args:
                lithology_3d_array (np.array of dim 3): lithology numeric (lithology class) identifiers
                dem_array_data (pandas.core.frame.DataFrame): dem data
        """
        dem_z = dem_array_data['dem_array']
        for i in range(lithology_3d_array.shape[0]):
            for j in range(lithology_3d_array.shape[1]):
                for k in range(lithology_3d_array.shape[2]):
                    height = k * (self.ahd_max - self.ahd_min) / lithology_3d_array.shape[2] + self.ahd_min
                    if height >= dem_z[i][j]:
                        for tmp in range(k, lithology_3d_array.shape[2]):
                            lithology_3d_array[i][j][tmp] = None
                        break
        return lithology_3d_array

    def vag_clean(self, lithology_3d_array, dem_array_data):
        """Simply process data that exceeds limits（we suppose that the lithology would not higher than the ground surface）,
        not accurate but faster

            Args:
                lithology_3d_array (np.array of dim 3): lithology numeric (lithology class) identifiers
                dem_array_data (pandas.core.frame.DataFrame: dem data
        """
        dem_z = dem_array_data['dem_array']
        for i in range(lithology_3d_array.shape[0]):
            for j in range(lithology_3d_array.shape[1]):
                k = int(dem_z[i][j] - self.ahd_min)
                for tep in range(k, lithology_3d_array.shape[2]):
                    lithology_3d_array[i][j][tep] = None
        return lithology_3d_array

    def build_layer_mesh(self, lithology_3d_array, height_adjustment_factor, layer_from, layer_to):
        """Build a 3D mesh of selected lithology class codes by binary bining cells. Use filter to select aim layers.

            Args:
                lithology_3d_array (np.array of dim 3): lithology numeric (lithology class) identifiers
                height_adjustment_factor (int): height scale factor
                layer_from (float): set the begin number of layers
                layer_to (float): set the end number of layers
        """
        volume = pv.UniformGrid()
        volume.dimensions = np.array(lithology_3d_array.shape)
        volume.origin = (self.dem_x_min, self.dem_y_min, self.ahd_min * height_adjustment_factor)
        x_label = (self.dem_x_max - self.dem_x_min) / lithology_3d_array.shape[0]
        y_label = (self.dem_y_max - self.dem_y_min) / lithology_3d_array.shape[1]
        z_label = (self.ahd_max - self.ahd_min) * height_adjustment_factor / lithology_3d_array.shape[2]
        volume.spacing = (x_label, y_label, z_label)
        volume.point_arrays["Lithology"] = lithology_3d_array.flatten('F')
        volume.set_active_scalar("Lithology")
        threshed = volume.threshold([layer_from, layer_to])
        return threshed

    def extract_single_lithology_class_3d(lithology_3d_classes, class_value):
        """Transform a 3D volume of lithology class codes by binary bining cells as being either of a class value or
        other. Preprocessing primarily for 3D visualisation for pyvista(not use in the sample ).

            Args:
                lithology_3d_classes (np.array of dim 3): lithology numeric (lithology class) identifiers
                class_value (float): class code of interest
        """
        single_litho = np.copy(lithology_3d_classes)
        other_value = None
        single_litho[(single_litho != class_value)] = other_value
        # We burn the edges of the volume, as I suspect this is necessary to have a more intuitive viz (otherwuse non
        # closed volumes)
        single_litho[0, :, :] = other_value
        single_litho[-1, :, :] = other_value
        single_litho[:, 0, :] = other_value
        single_litho[:, -1, :] = other_value
        single_litho[:, :, 0] = other_value
        single_litho[:, :, -1] = other_value
        return single_litho
