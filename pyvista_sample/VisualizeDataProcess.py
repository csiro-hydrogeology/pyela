import os
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
        self.grid_res = ''
        self.scalar_prop_name = "litho_num"

    def drill_file_read(self, file_path):

        """Read drill data file
            Args:
                file_path (str): drill data file path

            Returns:
                df(pandas.core.frame.DataFrame)
        """
        df = pd.read_pickle(file_path)
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
        handle.close()
        return dem_array_data

    def drill_data_initial(self, drill_data, depth_from_ahd=DEPTH_FROM_AHD_COL, depth_to_ahd=DEPTH_TO_AHD_COL):
        """initial class variables and clean drilling data
            Args:
                drill_data (pandas.core.frame.DataFrame): original drilling data
                depth_from_ahd(str):set the column name of depth from AHD, default DEPTH_FROM_AHD_COL
                depth_to_ahd(str):set the column name of depth to AHD, default DEPTH_TO_AHD_COL

            Returns:
                drill_data(pandas.core.frame.DataFrame)
        """
        self.ahd_max = drill_data[depth_from_ahd].max()
        self.ahd_min = drill_data[depth_to_ahd].min()
        # clean the invalid data
        return drill_data.dropna(subset=[depth_to_ahd, depth_from_ahd])

    def dem_data_initial(self, dem_array_data, dem_bounds='bounds', dem_grid_res='grid_res'):
        """initial class variables and clean dem data
            Args:
                dem_array_data (pandas.core.frame.DataFrame): original dem data
                dem_bounds(str): set bounds column name according to dem files
                dem_grid_res(str): set grid_res column name according to dem files

            Returns:
                dem_array_data(pandas.core.frame.DataFrame)
        """
        self.dem_x_min, self.dem_x_max, self.dem_y_min, self.dem_y_max = dem_array_data[dem_bounds]
        self.grid_res = dem_array_data[dem_grid_res]
        return dem_array_data

    def drill_data_process(self, drill_data, height_adjustment_factor=20, depth_from_ahd=DEPTH_FROM_AHD_COL,
                           depth_to_ahd=DEPTH_TO_AHD_COL, drill_east='Easting', drill_north='Northing',
                           boreID='BoreID', prime_lithology='Lithology_1_num', min_tube_radius=10):
        """The whole data process from drill data to PolyData dictionary

            Args:
                drill_data(pandas.core.frame.DataFrame): original drilling data
                height_adjustment_factor (int): Height scaling factor, default 20.
                depth_from_ahd(str):set the column name of depth from AHD, default DEPTH_FROM_AHD_COL
                depth_to_ahd(str):set the column name of depth to AHD, default DEPTH_TO_AHD_COL
                drill_east(str):set the column name of  point's x location in drilling data, default "Easting"
                drill_north(str):set the column name of  point's y's location in drilling data, default "Northing"
                boreID(str):set the column name of bore hole ID,default "BoreID"
                prime_lithology(str):set the prime lithology column name
                min_tube_radius(int):set the min radius of borehole tube

            Returns:
                lines_dict(dict): PolyData dictionary.
        """
        # data = self.drill_file_read(file_path, depth_from_ahd, depth_to_ahd)
        fixed_data = self.drill_data_initial(drill_data, depth_from_ahd, depth_to_ahd)
        data = self.add_scaled_height_column(fixed_data, height_adjustment_factor, depth_from_ahd, depth_to_ahd)
        well_dict = self.build_well_dict(data, boreID)
        # = self.add_missing_height_data(well_dict)
        point_dict = self.build_points_dict(well_dict, drill_east, drill_north)
        lines_dict = self.point_to_lines_dict(point_dict)
        lines_dict = self.add_lithology_based_scalar(well_dict, lines_dict, prime_lithology, min_tube_radius)
        return lines_dict

    def dem_data_process(self, dem_array_data, height_adjustment_factor, dem_mesh_xy='mesh_xy', dem_arrays='dem_array',
                         dem_bounds='bounds', dem_grid_res='grid_res'):
        """The whole data process from dem data to pv.StructuredGrid

            Args:
                dem_array_data (pandas.core.frame.DataFrame): original dem data
                height_adjustment_factor (int): Height scaling factor, default 20 .
                dem_mesh_xy(str): set mesh_xy column name according to dem files
                dem_arrays(str): set dem array column name according to dem files
                dem_bounds(str): set bounds column name according to dem files
                dem_grid_res(str): set grid_res column name according to dem files

            Returns:
                Grid(pyvista.core.pointset.StructuredGrid)

        """
        dem_array_data = self.dem_data_initial(dem_array_data, dem_bounds, dem_grid_res)
        xx, yy = dem_array_data[dem_mesh_xy]
        dem_array = dem_array_data[dem_arrays]
        grid = pv.StructuredGrid(xx, yy, dem_array * height_adjustment_factor)
        return grid

    def lithology_layer_process(self, drill_data, dem_array_data, storage_file_name, height_adjustment_factor=20,
                                layer_from=0, layer_to=0, dem_bounds='bounds', dem_grid_res='grid_res',
                                dem_mesh_xy='mesh_xy', drill_east='Easting', drill_north='Northing',
                                dem_arrays='dem_array', depth_from_ahd=DEPTH_FROM_AHD_COL,
                                depth_to_ahd=DEPTH_TO_AHD_COL):
        """add points lithology type, expands lines to tube based on lithology number
            Args:
                drill_data(pandas.core.frame.DataFrame): original drilling data
                dem_array_data (pandas.core.frame.DataFrame): original dem data
                storage_file_name(str): set the name of the save path for testing sample's
                                        lithology classification array
                height_adjustment_factor(int): height scala factor
                layer_from (float): set the begin number of layers
                layer_to (float): set the end number of layers
                dem_bounds(str): set bounds column name according to dem files
                dem_grid_res(str): set grid_res column name according to dem files
                dem_mesh_xy(str): set mesh_xy column name according to dem files
                drill_east(str):set the column name of  point's x location in drilling data, default "Easting"
                drill_north(str):set the column name of  point's y's location in drilling data, default "Northing"
                dem_arrays(str): set dem array column name according to dem files
                depth_from_ahd(str):set the column name of depth from AHD, default DEPTH_FROM_AHD_COL
                depth_to_ahd(str):set the column name of depth to AHD, default DEPTH_TO_AHD_COL


            Returns:
                layer_mesh(pyvista.core.pointset.UnstructuredGrid): layer mesh for display use
        """

        # drill_data = self.drill_file_read(drill_file_path, depth_from_ahd, depth_to_ahd)
        # dem_array_data = self.dem_file_read(dem_file_path, dem_bounds, dem_grid_res)
        path = os.path.join(storage_file_name, "lithology_3d_array.pkl")
        try:
            with open(path, 'rb') as handle:
                lithology_3d_array = pickle.load(handle)
            handle.close()
        except:
            drill_data = self.drill_data_initial(drill_data, depth_from_ahd, depth_to_ahd)
            dem_array_data = self.dem_data_initial(dem_array_data, dem_bounds, dem_grid_res)
            lithology_3d_array = self.build_layer_data(drill_data, dem_array_data, dem_mesh_xy, drill_east, drill_north)
            lithology_3d_array = self.clean_over_bound_data(lithology_3d_array, dem_array_data, dem_arrays)
            # lithology_3d_array = self.vag_clean(lithology_3d_array, dem_array_data)
            folder = os.path.exists(path)
            if not folder:
                os.makedirs(storage_file_name)
            with open(path, "wb") as cf:
                pickle.dump(lithology_3d_array, cf)
            cf.close()

        layer_mesh = self.build_layer_mesh(lithology_3d_array, height_adjustment_factor, layer_from, layer_to)
        return layer_mesh

    def add_scaled_height_column(self, data, height_adjustment_factor, depth_from_ahd=DEPTH_FROM_AHD_COL,
                                 depth_to_ahd=DEPTH_TO_AHD_COL):
        """Add scaled height columns to data frame
            Args:
                data (pandas.core.frame.DataFrame):original data
                height_adjustment_factor (int): Height scaling factor.
                depth_from_ahd(str):set the column name of depth from AHD, default DEPTH_FROM_AHD_COL
                depth_to_ahd(str):set the column name of depth to AHD, default DEPTH_TO_AHD_COL
            Returns:
                data(pandas.core.frame.DataFrame): modified data
        """
        # scaled_from_height_colname = 'scaled_from_height'
        data.loc[:, self.scaled_from_height_colname] = data[depth_from_ahd].values * height_adjustment_factor
        # scaled_to_height_colname = 'scaled_to_height'
        data.loc[:, self.scaled_to_height_colname] = data[depth_to_ahd].values * height_adjustment_factor
        return data

    def build_well_dict(self, data, boreID='BoreID'):
        """build dictionary according to BoreID
            Args:
                data (pandas.core.frame.DataFrame):original data
                boreID(str):set the column name of bore hole ID,default "BoreID"
            Returns:
                well_dict(dict()): wells dictionary
        """
        data.loc[:, 'name'] = data.loc[:, boreID].values.astype(str)
        wells = data.name.unique()
        well_dict = {}
        for well in wells:
            well_dict["{0}".format(well)] = data[data.name == well]
        return well_dict

    # def add_missing_height_data(self, well_dict):
    #     """Add the smallest height_to data to height_from data (len(well_dict[i])+1)
    #         Args:
    #             well_dict(dict()): original dictionary
    #         Returns:
    #             well_dict(dict()): modified dictionary
    #     """
    # bad_well = []
    # for well in well_dict.keys():
    #    origin_well_df = well_dict.get(well)
    #    after_well_df = origin_well_df.copy()
    #    add_index = origin_well_df[self.scaled_to_height_colname].idxmin()
    #    if np.isnan(add_index):
    #        bad_well.append(well)
    #        continue
    #    line = origin_well_df.loc[add_index].copy()
    #    line.scaled_from_height = line.scaled_to_height
    #    line = line.to_frame()
    #    temp = []
    #    for value in line.values:
    #        if value[0]:
    #            temp.append(value[0])
    #        else:
    #            temp.append(0)
    #    after_well_df.loc["new"] = temp
    #    well_dict[well] = after_well_df
    # for i in range(len(bad_well)):
    #    well_dict.pop(bad_well[i])
    # return well_dict

    def build_points_dict(self, well_dict, drill_east="Easting", drill_north="Northing"):
        """build points dictionary from wells dictionary
            Args:
                well_dict(dict()): wells dictionary
                drill_east(str):set the column name of  point's x location in drilling data, default "Easting"
                drill_north(str):set the column name of  point's y's location in drilling data, default "Northing"
            Returns:
                points_dict(dict()): zip points axis for points
        """
        c = np.concatenate
        points_dict = {}
        for points in well_dict:
            e = well_dict[points][drill_east].values
            n = well_dict[points][drill_north].values
            points_dict["{0}".format(points)] = np.array(
                list(
                    zip(
                        c((e, e)),
                        c((n, n)),
                        c((
                            well_dict[points][self.scaled_from_height_colname].values,
                            well_dict[points][self.scaled_to_height_colname].values + 1.0))
                    )
                )
            )
        return points_dict

    def point_to_lines_dict(self, points_dict):
        """build lines dictionary from points dictionary
            Args:
                points_dict(dict()): points dictionary
            Returns:
                lines_dict(dict()): build lines between same well points
        """
        lines_dict = {}
        for bore_id in points_dict:
            poly = PVGeo.points_to_poly_data(points_dict[bore_id])
            lines_dict["{0}".format(bore_id)] = PVGeo.filters.AddCellConnToPoints(nearest_nbr=True).apply(poly)
            # notice that the building of the lines need to follow the nearest neighbourhood search
        return lines_dict

    def add_lithology_based_scalar(self, well_dict, lines_dict, prime_lithology='Lithology_1_num', min_tube_radius=10):
        """add points lithology type, expands lines to tube based on lithology number
            Args:
                well_dict(dict()): wells dictionary
                lines_dict(dict()):lines dictionary
                prime_lithology(str):set the prime lithology column name
                min_tube_radius(int): set the min radius of borehole tube
            Returns:
                lines_dict(dict()): with new attribute "GR" which represent lithology number, and expanded to tube.
        """
        lines_dict_tmp = {}
        for bore_id in lines_dict:
            try:
                vals = well_dict[bore_id][prime_lithology].values
                bore_vis = lines_dict[bore_id]
                bore_vis[self.scalar_prop_name] = np.concatenate((vals, vals))  # tops then bottoms of cylinders.
                bore_vis.tube(radius=min_tube_radius, scalars=None, inplace=True)
                # lines_dict[bore_id].tube(radius=10, scalars=dp.scalar_prop_name, inplace=True)
            except Exception as e:
                raise Exception("Lithology attribute processed for visualisation failed for bore ID %s" % (bore_id))
            if len(vals) > 0:
                lines_dict_tmp[bore_id] = lines_dict[bore_id]
        lines_dict = lines_dict_tmp
        return lines_dict

    def build_layer_data(self, drill_data, dem_array_data, dem_mesh_xy='mesh_xy', drill_east='Easting',
                         drill_north='Northing'):
        """get the layer data from the function contains in ela
            Args:
                drill_data (pandas.core.frame.DataFrame): drill data
                dem_array_data (pandas.core.frame.DataFrame): dem data
                dem_mesh_xy(str): set mesh_xy column name according to dem files
                drill_east(str):set the column name of  point's x location in drilling data, default "Easting"
                drill_north(str):set the column name of  point's y's location in drilling data, default "Northing"
        """
        n_neighbours = 10
        xg, yg = dem_array_data[dem_mesh_xy]
        m = create_meshgrid_cartesian(self.dem_x_min, self.dem_x_max, self.dem_y_min, self.dem_y_max, self.grid_res)
        z_coords = np.arange(self.ahd_min, self.ahd_max, 1)
        dim_x, dim_y = xg.shape
        dim_z = len(z_coords)
        dims = (dim_x, dim_y, dim_z)
        lithology_3d_array = np.empty(dims)
        gi = GridInterpolation(easting_col=drill_east, northing_col=drill_north)
        gi.interpolate_volume(lithology_3d_array, drill_data, PRIMARY_LITHO_NUM_COL, z_coords, n_neighbours, m)
        return lithology_3d_array

    def clean_over_bound_data(self, lithology_3d_array, dem_array_data, dem_arrays='dem_array'):
        """accurate process data that exceeds limits
        （we suppose that the lithology would not higher than the ground surface),
        accurate but slower
            Args:
                lithology_3d_array (np.array of dim 3): lithology numeric (lithology class) identifiers
                dem_array_data (pandas.core.frame.DataFrame): dem data
                dem_arrays(str): set dem array column name according to dem files
        """
        dem_z = dem_array_data[dem_arrays]
        for i in range(lithology_3d_array.shape[0]):
            for j in range(lithology_3d_array.shape[1]):
                if np.isnan(dem_z[i][j]):
                    lithology_3d_array[i][j] = None
                    continue
                for k in range(lithology_3d_array.shape[2]):
                    height = k * (self.ahd_max - self.ahd_min) / lithology_3d_array.shape[2] + self.ahd_min
                    if height >= dem_z[i][j]:
                        for tmp in range(k, lithology_3d_array.shape[2]):
                            lithology_3d_array[i][j][tmp] = None
                        break
        return lithology_3d_array

    def vag_clean(self, lithology_3d_array, dem_array_data, dem_arrays='dem_array'):
        """Simply process data that exceeds limits（we suppose that the lithology would not higher than the ground surface）,
        not accurate but faster

            Args:
                lithology_3d_array (np.array of dim 3): lithology numeric (lithology class) identifiers
                dem_array_data (pandas.core.frame.DataFrame: dem data
                dem_arrays(str): set dem array column name according to dem files
        """
        dem_z = dem_array_data[dem_arrays]
        for i in range(1, lithology_3d_array.shape[0]):
            for j in range(1, lithology_3d_array.shape[1]):
                if np.isnan(dem_z[i][j]):
                    k = 0
                else:
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

    # def exist_3d_lithology(self):

    def extract_single_lithology_class_3d(self, lithology_3d_classes, class_value):
        """Transform a 3D volume of lithology class codes by binary bining cells as being either of a class value or
        other. Preprocessing primarily for 3D visualisation for pyvista(not use in the sample ).

            Args:
                lithology_3d_classes (np.array of dim 3): lithology numeric (lithology class) identifiers
                class_value (float): class code of interest
        """
        single_litho = np.copy(lithology_3d_classes)
        other_value = None
        single_litho[(single_litho != class_value)] = other_value
        # We burn the edges of the volume, as I suspect this is necessary to have a more intuitive viz (otherwise non
        # closed volumes)
        single_litho[0, :, :] = other_value
        single_litho[-1, :, :] = other_value
        single_litho[:, 0, :] = other_value
        single_litho[:, -1, :] = other_value
        single_litho[:, :, 0] = other_value
        single_litho[:, :, -1] = other_value
        return single_litho
    # burn_volume
