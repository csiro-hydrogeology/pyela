from __future__ import print_function # anything else?

import sys
import pandas as pd
import numpy as np
from sklearn import neighbors

import geopandas as gpd

from ela.textproc import EASTING_COL, NORTHING_COL, DEPTH_FROM_AHD_COL, DEPTH_FROM_COL, DEPTH_TO_AHD_COL, DEPTH_TO_COL, PRIMARY_LITHO_COL, PRIMARY_LITHO_NUM_COL, SECONDARY_LITHO_COL, GEOMETRY_COL
from ela.spatial import *

KNN_WEIGHTING = 'distance'
# 0=sand
# 1=sandstone 
# 2=clay
# 3=limestone
# 4=shale
# 5=basalt
# 6=coffee rock
LITHOLOGIES = ['sand','sandstone','clay','limestone','shale','basalt','coffee']


def to_litho_class_num(lithology, kv):
    """Get a numeric code for a lithology, or NaN if not in the dictionary mapping lithologies to numeric code
            
        Args:
            lithology (str): Name of the lithology
            kv (dict[str,float]): lithologies keywords to numeric code
    """
    if lithology in kv.keys():
        return kv[lithology]
    else:
        return np.nan

def v_to_litho_class_num(lithologies, kv):
    """Get numeric codes for lithologies, or NaN if not in the dictionary mapping lithologies to numeric code
            
        Args:
            lithologies (iterable of str): Name of the lithologies
            kv (dict[str,float]): lithologies keywords to numeric code
    """
    return np.array([to_litho_class_num(x, kv) for x in lithologies])

def create_numeric_classes(lithologies):
    """Creates a dictionary mapping lithologies to numeric code
            
        Args:
            lithologies (iterable of str): Name of the lithologies
    """
    my_lithologies_numclasses = dict([(lithologies[i], i) for i in range(len(lithologies))])
    return my_lithologies_numclasses


class ClassMapper:
    """Hold information about and perform lithology and hydraulic conductivity classification

    Attributes:
        lithology_names (iterable of str): Name of the lithologies
        mapping (dict): dictionary where keys are primary+secondary lithologies ('sand/clay') and values are numeric codes for e.g. hydraulic conductivities
        litho_numeric_mapper (np array): numeric mapper from primary+secondary lithologies to hydraulic conductivities
    """
    def __init__(self, mapping, lithology_names):
        """initialize this with a coordinate reference system object and an affine transform. See rasterio.
        
        Args:
            lithology_names (iterable of str): Name of the lithologies
            mapping (dict): dictionary where keys are primary+secondary lithologies ('sand/clay') and values are numeric codes for e.g. hydraulic conductivities
        """
        self.mapping = mapping
        self.litho_numeric_mapper = np.empty((len(lithology_names), len(lithology_names)))
        for i in np.arange(0, len(lithology_names), 1):
            for j in np.arange(0, len(lithology_names), 1):
                self.litho_numeric_mapper[i,j] = self.class_code(lithology_names[i], lithology_names[j])
        self.lithology_names = lithology_names
    @staticmethod
    def create_full_litho_desc(df):
        """Create strings identifying primary+secondary lithologies, used as keys in classification operations

            Args:
                df (pandas data frame): bore lithology data  with columns named PRIMARY_LITHO_COL and SECONDARY_LITHO_COL
        
            Returns:
                a list of strings, compound primary+optional_secondary lithology descriptions e.g. 'sand/clay', 'loam/'
        """
        p1 = df[PRIMARY_LITHO_COL].values
        p2 = df[SECONDARY_LITHO_COL].values
        return ['/'.join([p1[i], p2[i]]) for i in range(len(p1))]
    def _mapping_class(self, litho_class):
        keys = self.mapping.keys()
        if litho_class in keys:
            return self.mapping[litho_class]
        else:
            return np.nan
    def _to_int(self, f):
        if np.isnan(f): return f
        return int(round(f))
    def map_classes(self, litho_classes):
        """Get the numeric class for primary+secondary lithologies

            Args:
                litho_classes (iterable of str): one or more strings e.g. 'sand/clay'
        
            Returns:
                list of numeric codes
        """
        return [self._mapping_class(x) for x in litho_classes]
    def litho_class_label(self, primary_litho_class, secondary_litho_class):
        """Get the string identifier for a set of primary+secondary lithologies

            Args:
                primary_litho_class (str, float or int): primary lithology name or numeric (lithology class) identifier
                secondary_litho_class (str, float or int): primary lithology name or numeric (lithology class) identifier
        
            Returns:
                string, lithologies key such as 'sand/clay'
        """
        if isinstance(primary_litho_class, float):
            if np.isnan(primary_litho_class): return np.nan
            primary_litho_class = self._to_int(primary_litho_class)
        if isinstance(secondary_litho_class, float):
            if np.isnan(secondary_litho_class): 
                secondary_litho_class = ''
            else:
                secondary_litho_class = self._to_int(secondary_litho_class)
        if isinstance(primary_litho_class, int): primary_litho_class = self.lithology_names[primary_litho_class]
        if isinstance(secondary_litho_class, int): secondary_litho_class = self.lithology_names[secondary_litho_class]
        litho_class = '/'.join([primary_litho_class, secondary_litho_class])
        return litho_class
    def class_code(self, primary_litho_class, secondary_litho_class):
        """Get the mapping class code (e.g. hydraulic conductivity) for a set of primary+secondary lithologies

            Args:
                primary_litho_class (str, float or int): primary lithology name or numeric (lithology class) identifier
                secondary_litho_class (str, float or int): primary lithology name or numeric (lithology class) identifier
        
            Returns:
                numeric, numeric code of the mapped class for this  primary+secondary lithologies
        """
        return self._mapping_class(self.litho_class_label(primary_litho_class, secondary_litho_class))
    def bivariate_mapper(self, primary_litho_code, secondary_litho_code):
        """Get the mapping class code (e.g. hydraulic conductivity) for a set of primary+secondary lithologies

            Args:
                primary_litho_class (float): primary lithology numeric (lithology class) identifier
                secondary_litho_class (float): primary lithology numeric (lithology class) identifier
        
            Returns:
                numeric, numeric code of the mapped class for this  primary+secondary lithologies
        """
        if np.isnan(primary_litho_code):
            return np.nan
        if np.isnan(secondary_litho_code): 
            return self.litho_numeric_mapper[self._to_int(primary_litho_code), self._to_int(primary_litho_code)]
        return self.litho_numeric_mapper[self._to_int(primary_litho_code), self._to_int(secondary_litho_code)]
    def map_classes(self, primary_lithology_3d_array, secondary_lithology_3d_array):
        """(compute intensive) Get the mapping class codes (e.g. hydraulic conductivity) for grids of primary+secondary lithologies

            Args:
                primary_lithology_3d_array (np.array of dim 3): primary lithology numeric (lithology class) identifiers
                secondary_lithology_3d_array (np.array of dim 3): primary lithology numeric (lithology class) identifiers
        
            Returns:
                (np.array of dim 3): mapped numeric identifiers, e.g. as of hydraulic conductivities
        """
        three_k_classes = primary_lithology_3d_array.copy()
        dim_x,dim_y,dim_z = three_k_classes.shape
        for i in np.arange(0, dim_x, 1):
            for j in np.arange(0, dim_y, 1):
                for k in np.arange(0, dim_z, 1):
                    three_k_classes[i,j,k] = self.bivariate_mapper(primary_lithology_3d_array[i,j,k], secondary_lithology_3d_array[i,j,k])    
        return three_k_classes
    def get_frequencies(self, mask_2d, primary_lithology_3d_array, secondary_lithology_3d_array):
        """Get the frequencies of primary+secondary for a set of x/y coordinates and all Z values in 3d lithologies

            Args:
                mask_2d (np.array of dim 2): mask to apply to the x-y dimensions of the other arguments
                primary_lithology_3d_array (np.array of dim 3): primary lithology numeric (lithology class) identifiers
                secondary_lithology_3d_array (np.array of dim 3): primary lithology numeric (lithology class) identifiers
        
            Returns:
                (np.array of dim 2): counts of primary/secondary lithology occurrences.
        """
        result = np.zeros([len(self.lithology_names), len(self.lithology_names)])
        ## TODO should check on dimensions...
        dim_x,dim_y,dim_z = secondary_lithology_3d_array.shape
        for i in np.arange(0, dim_x, 1):
            for j in np.arange(0, dim_y, 1):
                if mask_2d[i,j]:
                    for k in np.arange(0, dim_z, 1):
                        prim_litho_ind = self._to_int(primary_lithology_3d_array[i,j,k])
                        if np.isnan(prim_litho_ind) == False:
                            sec_litho_ind = self._to_int(secondary_lithology_3d_array[i,j,k])
                            if np.isnan(sec_litho_ind): 
                                sec_litho_ind = prim_litho_ind
                            result[prim_litho_ind,sec_litho_ind] = result[prim_litho_ind,sec_litho_ind] + 1
        return result
    def data_frame_frequencies(self, freq_table):
        """Get the frequencies of primary+secondary as a data frame, typically from the output of get_frequencies

            Args:
                freq_table (np.array of dim 2): counts of primary/secondary lithology occurrences.
        
            Returns:
                (pandas data frame): counts of primary/secondary lithology occurrences.
        """
        ## TODO should check on dimensions...
        x = [(self.litho_class_label(i, j), freq_table[i,j]) for i in range(len(self.lithology_names)) for j in range(len(self.lithology_names))]
        return pd.DataFrame(x, columns=["token","frequency"])


def lithologydata_slice_depth(df, slice_depth, depth_from_colname=DEPTH_FROM_AHD_COL, depth_to_colname=DEPTH_TO_AHD_COL):
    """
    Subset data frame with entries at a specified AHD coordinate

        Args:
            df (pandas data frame): bore lithology data  
            slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations 
    
        Returns:
            a (view of a) data frame, a subset of the input data frame, 
            entries intersecting with the specified slice depth
    """
    df_slice=df.loc[(df[depth_from_colname] >= slice_depth) & (df[depth_to_colname] <= slice_depth)]
    return df_slice

# The following was spurred by trying to get more data in KNN cross-validation, but this may be dubious method to increase the data pool. Park.
# def get_lithology_observations_between(df, bottom_ahd, top_ahd, column_name ):
#     """
#     Subset data frame with entries at a specified AHD coordinate, and with valid lithology information.

#         Args:
#             df (pandas data frame): bore lithology data  
#             bottom_ahd (float): bottom AHD coordinate of the slice to subset
#             top_ahd (float): top AHD coordinate of the slice 
#             column_name (str): name of the column with string information to use to strip entries with missing lithology information
    
#         Returns:
#             a (view of a) data frame; a subset of the input data frame, 
#             entries intersecting with the specified slice depth
#     """
#     depth_from_colname=DEPTH_FROM_AHD_COL
#     depth_to_colname=DEPTH_TO_AHD_COL
#     df_slice=df.loc[(df[depth_from_colname] >= top_ahd) & (df[depth_to_colname] <= slice_depth)] # CAREFUL HERE about order and criteria... trickier than 2D slicing.
#     df_1=df_slice[np.isnan(df_slice[column_name]) == False]
#     return df_1

def get_lithology_observations_for_depth(df, slice_depth, column_name ):
    """
    Subset data frame with entries at a specified AHD coordinate, and with valid lithology information.

        Args:
            df (pandas data frame): bore lithology data  
            slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations 
            column_name (str): name of the column with string information to use to strip entries with missing lithology information
    
        Returns:
            a (view of a) data frame; a subset of the input data frame, 
            entries intersecting with the specified slice depth
    """
    df_slice=lithologydata_slice_depth(df, slice_depth)
    df_1=df_slice[np.isnan(df_slice[column_name]) == False]
    return df_1

def extract_bore_class_num(bore_log_df, column_name):
    """
    Gets the columns easting, northing, primary lithology class number, AHD depth 'from' and 'to' from a bore data log
    """
    xx = bore_log_df[EASTING_COL].values
    yy = bore_log_df[NORTHING_COL].values
    ss = bore_log_df[column_name].values
    zz_from = bore_log_df[DEPTH_FROM_AHD_COL].values
    zz_to = bore_log_df[DEPTH_TO_AHD_COL].values
    return xx, yy, zz_from, zz_to, ss

def make_training_set(observations, column_name):
    # X = observations.as_matrix(columns=[EASTING_COL, NORTHING_COL])
    X = observations[[EASTING_COL, NORTHING_COL]].values
    y = np.array(observations[column_name])
    #NOTE: should I also do e.g.:
    #shuffle_index = np.random.permutation(len(y))
    #X, y = X[shuffle_index], y[shuffle_index]   
    return (X, y)

def get_knn_model(df, column_name, slice_depth, n_neighbours):
    """Train a K-nearest neighbours model for a given plane 

    Args:
        df (data frame): 
        column_name (str): 
        slice_depth (numeric): 
        n_neighbours (int): 

    Returns:
        KNeighborsClassifier: trained classifier.
    """
    df_1 = get_lithology_observations_for_depth(df, slice_depth, column_name)
    X, y = make_training_set(df_1, column_name)
    if n_neighbours > len(df_1):
        return None
    else:
        knn = neighbors.KNeighborsClassifier(n_neighbours, weights = KNN_WEIGHTING).fit(X, y)
        return knn

def interpolate_over_meshgrid(predicting_algorithm, mesh_grid):
    """
    Interpolate lithology data

    :predicting_algorithm: algorithm such as KNN
    :type: algorithm with a predict method.
        
    :mesh_grid: coordinate matrices to interpolate over (numpy.meshgrid)
    :type: tuple
    
    :return: predicted values over the grid
    :rtype: numpy array

    """
    xx, yy = mesh_grid
    if predicting_algorithm is None:
        # the training set was too small and prediction cannot be made (odd that scikit would have let us train still)
        predicted = np.empty(xx.shape)
        predicted[:] = np.nan # np.empty should have done that already, but, no...
    else:
        predicted = predicting_algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
        predicted = predicted.reshape(xx.shape)
    return predicted

def interpolate_lithologydata_slice_depth(df, column_name, slice_depth, n_neighbours, mesh_grid):
    """
    Interpolate lithology data

    :df: bore lithology data  
    :type: pandas data frame 
    
    :slice_depth: AHD coordinate at which to slice the data frame for lithology observations 
    :type: double  
    
    :n_neighbours: Number of neighbors to pass to KNeighborsClassifier
    :type: integer
    
    :mesh_grid: coordinate matrices to interpolate over (numpy.meshgrid)
    :type: tuple
    
    :return: predicted values over the grid
    :rtype: numpy array

    """
    knn = get_knn_model(df, column_name, slice_depth, n_neighbours)
    return interpolate_over_meshgrid(knn, mesh_grid)

def interpolate_lithologydata_slice_depth_bbox(df, column_name, slice_depth, n_neighbours, geo_pd, grid_res = 100):
    """
    Interpolate lithology data

    :df: bore lithology data  
    :type: pandas data frame 
    
    :slice_depth: AHD coordinate at which to slice the data frame for lithology observations 
    :type: double  
    
    :n_neighbours: Number of neighbors to pass to KNeighborsClassifier
    :type: integer
    
    :geo_pd: vector spatial data to get bounds of interest (box)
    :type: 
    
    :grid_res: grid resolution in m for x and y.
    :type: integer
    
    :return: predicted values over the grid
    :rtype: numpy array

    """
    mesh_grid = create_meshgrid(geo_pd, grid_res)
    return interpolate_lithologydata_slice_depth(df, column_name, slice_depth, n_neighbours, mesh_grid)


def class_probability_estimates_depth(df, column_name, slice_depth, n_neighbours, mesh_grid, func_training_set=None):
    """Subset data frame with entries at a specified AHD coordinate

        Args:
            df (pandas data frame): bore lithology data  
            slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations
            n_neighbours (int): number of nearest neighbours 
            mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)
            func_training_set (callable):  a function to processing the training set (e.g. completing dummy with dummy classes, other not present in the trainining set)

        Returns:
            a list of numpy arrays, shaped like the meshgrid.
    """
    df_1 = get_lithology_observations_for_depth(df, slice_depth, column_name)
    X, y = make_training_set(df_1, column_name)
    if not (func_training_set is None):
        X, y = func_training_set(X, y)
    knn = neighbors.KNeighborsClassifier(n_neighbours, weights = KNN_WEIGHTING).fit(X, y)
    xx, yy = mesh_grid
    class_prob = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    n_classes = class_prob.shape[1]
    probs = []
    for i in range(n_classes):
        p = class_prob[:,i].reshape(xx.shape)
        probs.append(p)
    return probs

def class_probability_estimates_depth_bbox(df, column_name, slice_depth, n_neighbours, geo_pd, grid_res = 100, func_training_set=None):
    mesh_grid = create_meshgrid(geo_pd, grid_res)
    return class_probability_estimates_depth(df, column_name, slice_depth, n_neighbours, mesh_grid, func_training_set)

def interpolate_volume(volume, df, column_name, z_ahd_coords, n_neighbours, mesh_grid):
    dim_x,dim_y = mesh_grid[0].shape
    dim_z = len(z_ahd_coords)
    if volume.shape[0] != dim_x or volume.shape[1] != dim_y or volume.shape[2] != dim_z:
        raise Error("Incompatible dimensions in arguments")
    for index,ahd_height in enumerate(z_ahd_coords):
        surface = interpolate_lithologydata_slice_depth(df, column_name, ahd_height, n_neighbours, mesh_grid)
        volume[:,:,index]=surface


def pad_training_set_functor(classes):
    ### NEED TO APPEND DUMMY DATA TO MAKE SURE ALL CLASSES ARE PRESENT IN EACH SLICE ###
    # 0=sand
    # 1=sandstone 
    # 2=clay
    # 3=limestone
    # 4=shale
    # 5=basalt
    # 6=coffee rock
    n = len(classes)
    def pad_training_set(X, y):
        dummy_EN=np.array([[0,0] for i in range(n)])
        dummy_targets=np.array(range(n))
        X=np.vstack((X,dummy_EN))
        y=np.append(y,dummy_targets)
        return (X, y)
    return pad_training_set


def get_lithology_classes_probabilities(lithologies, shape, df, column_name, z_ahd_coords, n_neighbours, mesh_grid):
    dim_x,dim_y,dim_z = shape
    vol_template=np.empty((dim_x,dim_y,dim_z))
    classprob_3d_arrays=[vol_template.copy() for i in lithologies]
    n_classes = len(lithologies)
    pad_training_set = pad_training_set_functor(lithologies)
    # iterate over all slices
    for z_index,ahd_height in enumerate(z_ahd_coords):
        result=class_probability_estimates_depth(df, column_name, ahd_height, n_neighbours, mesh_grid, func_training_set = pad_training_set)
        for i in range(n_classes):
            classprob_3d_arrays[i][:,:,z_index]=result[i]
    return classprob_3d_arrays


def extract_single_lithology_class_3d(lithology_3d_classes, class_value):
    """Transform a 3D volume of lithology class codes by binary bining cells as being either of a class value or other. Preprocessing primarily for 3D visualisation for Mayavi.

        Args:
            lithology_3d_classes (np.array of dim 3): lithology numeric (lithology class) identifiers
            class_value (float): class code of interest
    """
    single_litho = np.copy(lithology_3d_classes)
    other_value = class_value-1.0
    single_litho[(single_litho != class_value)] = other_value
    # We burn the edges of the volume, as I suspect this is necessary to have a more intuitive viz (otherwuse non closed volumes)
    single_litho[0,:,:] = other_value
    single_litho[-1,:,:] = other_value
    single_litho[:,0,:] = other_value
    single_litho[:,-1,:] = other_value
    single_litho[:,:,0] = other_value
    single_litho[:,:,-1] = other_value
    return single_litho
