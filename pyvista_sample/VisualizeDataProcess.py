import pickle
import PVGeo
import pyvista as pv
import pandas as pd
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
        grid_res = dem_array_data['grid_res']
        x_min, x_max, y_min, y_max = dem_array_data['bounds']
        xx, yy = dem_array_data['mesh_xy']
        dem_array = dem_array_data['dem_array']
        grid = pv.StructuredGrid(xx, yy, dem_array * height_adjustment_factor)
        return grid

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
