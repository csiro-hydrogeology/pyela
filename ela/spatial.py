import numpy as np
import pandas as pd

from ela.textproc import EASTING_COL, NORTHING_COL, DEPTH_FROM_AHD_COL, DEPTH_FROM_COL, DEPTH_TO_AHD_COL, DEPTH_TO_COL, BORE_ID_COL

KNN_WEIGHTING = 'distance'

from sklearn import neighbors


def read_raster_value(dem,band_1,easting,northing):
    """Read a value in a raster grid given easting/northing

    Args:
        dem (rasterio dataset): dem
        band_1 (numpy array): the band 1 from the DEM
        easting (float): easting value
        northing (float): northing value

    Returns:
        (float): the value of the band at the specified point
    """
    if (np.isnan(easting) or np.isnan(northing)):
        raise Exception("Easting and northing must not be NaN")
    row, col = dem.index(easting,northing)
    # dem.index seems to return floats and this causes a but to index its numpy array.
    # something used to work (who knows which package version soup) and does not anymore. At runtime. Dynamic typing...
    row = int(row)
    col = int(col)
    if (row < 0 or row >= band_1.shape[0] or col < 0 or col >= band_1.shape[1]):
        return np.nan
    else:
        return band_1[row,col]

def get_coords_from_gpd_shape(shp, colname='geometry', out_colnames = None):
    """Gets a dataframe of x/y geolocations out of a geopandas object

        Args:
            shp (GeoDataFrame): a geopandas GeoDataFrame
            colname (str): name of columns that has the geometry in the geopandas shapefile
            out_colnames (list of str): names of the added geolocation columns in the output data frame
        Returns:
            (DataFrame): a two column data frames, coordinates of all the points in the geodataframe

        Example:
            >>> from ela.doc.sampledata import *
            >>> from ela.spatial import get_coords_from_gpd_shape
            >>> dem ,bore_loc, litho_logs = sample_data()
            >>> df = litho_logs[[DEPTH_FROM_COL, DEPTH_TO_COL,TOP_ELEV_COL,BOTTOM_ELEV_COL,LITHO_DESC_COL,HYDRO_CODE_COL]]
            >>> geoloc = get_coords_from_gpd_shape(bore_loc, colname='geometry', out_colnames=[EASTING_COL, NORTHING_COL])
            >>> geoloc[HYDRO_CODE_COL] = bore_loc[HYDRO_CODE_COL]
            >>> geoloc.head()
            >>> # With this data frame we can perform two operations in one go: subsetting the lithology records to only the 640 bores of interest, and adding to the result the x/y geolocations to the data frame.
            >>> df = pd.merge(df, geoloc, how='inner', on=HYDRO_CODE_COL, sort=False, copy=True, indicator=False, validate=None)
    """
    out_colnames = out_colnames or ["x","y"]
    p = shp[[colname]]
    pts = p.values.flatten()
    c = [(pt.x, pt.y) for pt in pts]
    return pd.DataFrame(c, columns=out_colnames)

def get_unique_coordinates(easting, northing):
    """Gets the unique set of geolocated points

        Args:
            easting (iterable of floats): easting values
            northing (iterable of floats): northing values
        Returns:
            (numpy array): two dimensional nparray
    """
    grid_coords = np.empty( (len(easting), 2) )
    grid_coords[:,0] = easting
    grid_coords[:,1] = northing
    b = np.unique(grid_coords[:,0] + 1j * grid_coords[:,1])
    points = np.column_stack((b.real, b.imag))
    return points

class HeightDatumConverter:
    """
    Attributes:
        dfcn (GeospatialDataFrameColumnNames): Adapter to a dataframe column names for depth/height information.
        dem_raster (rasterio dataset): DEM raster
        data_grid (??): band 1 data in the raster
    """
    def __init__(self, dem_raster, easting_col=EASTING_COL, northing_col=NORTHING_COL, depth_from_ahd_col=DEPTH_FROM_AHD_COL, depth_to_ahd_col=DEPTH_TO_AHD_COL):
        """Initialize this with a coordinate reference system object and an affine transform. See rasterio.

            Args:
                easting_col (str): name of the data frame column for easting
                northing_col (str): name of the data frame column for northing
                depth_from_ahd_col (str): name of the data frame column for the height of the top of the soil column (ahd stands for for australian height datum, but not restricted)
                depth_to_ahd_col (str): name of the data frame column for the height of the bottom of the soil column (ahd stands for for australian height datum, but not restricted)
        """
        self.dfcn = GeospatialDataFrameColumnNames(easting_col, northing_col, depth_from_ahd_col, depth_to_ahd_col)
        self.dem_raster = dem_raster
        self.data_grid = self.dem_raster.read(1)

    def _raster_drill_df(self, df, easting_col, northing_col):
        x = df[easting_col].values
        y = df[northing_col].values
        v = np.empty_like(x, dtype=self.data_grid.dtype) # Try to fix https://github.com/jmp75/pyela/issues/2
        for i in range(len(x)):
            v[i] = read_raster_value(self.dem_raster, self.data_grid, x[i], y[i])
        return v

    def raster_drill_df(self, df):
        """Gets the DEM values for the easting/northing in a data frame

        Args:
            df (pandas.DataFrame): data frame of records with easting/northing information in columns compatible with this object

        Returns:
            (1D numpy array): the datum height values at the specified points
        """
        return self._raster_drill_df(df, self.dfcn.easting_col, self.dfcn.northing_col)

    def add_height(self, lithology_df,
        depth_from_col=DEPTH_FROM_COL, depth_to_col=DEPTH_TO_COL,
        depth_from_ahd_col=DEPTH_FROM_AHD_COL, depth_to_ahd_col=DEPTH_TO_AHD_COL,
        easting_col=EASTING_COL, northing_col=NORTHING_COL,
        drop_na=False):
        """Helper class to round lithology record classes to the nearest metre of depth

        Args:
            lithology_df:
            depth_from_col (str): Column name storing "from depth" information
            depth_to_col (str): Column name storing "to depth" information
            depth_from_ahd_col (str): Column name datum height information
            depth_to_ahd_col (str): Column name datum height information
            easting_col (str): Column name storing easting information
            northing_col (str): Column name storing northing information
            drop_na (bool): If true, entries with nissing datum heights are dropped

        Returns:
            (pandas.DataFrame):
        """
        df = lithology_df.copy(deep=True)
        nd = np.float32(self.dem_raster.nodata) # Try to fix https://github.com/jmp75/pyela/issues/2
        ahd = self._raster_drill_df(df, easting_col, northing_col)
        ahd[ahd==nd] = np.nan
        df[depth_from_ahd_col]=ahd-df[depth_from_col]
        df[depth_to_ahd_col]=ahd-df[depth_to_col]
        if drop_na:
            df = df[pd.notna(df[depth_to_ahd_col])]
        return df

    def raster_value_at(self, easting, northing):
        """Raster value for a given easting/northing

        Args:
            easting (float): easting value
            northing (float): northing value

        Returns:
            (float): the value of the band at the specified point
        """
        return read_raster_value(self.dem_raster, self.data_grid, easting, northing)

class DepthsRounding:
    """Helper class to round lithology record classes to the nearest metre of depth

    Attributes:
        depth_from_col (str): Column name  storing "from depth" information
        depth_to_col (str): Column name  storing "to depth" information
    """
    def __init__(self, depth_from_col=DEPTH_FROM_COL, depth_to_col=DEPTH_TO_COL):
        """Helper class to round lithology record classes to the nearest metre of depth

        Args:
            depth_from_col (str): Column name  storing "from depth" information
            depth_to_col (str): Column name  storing "to depth" information
        """
        self.depth_from_col = depth_from_col
        self.depth_to_col = depth_to_col

    def round_to_metre_depths(self, df, func=np.round, remove_collapsed=False):
        """Round lithology record classes to the nearest metre of depth

        Args:
            df (pandas data frame): bore lithology data
            func (callable): rounding function callable with a signature similar to `np.round`
            remove_collapsed (bool): should entries where depths from and to are equal be removed from the result
        Returns:
            (pandas dataframe): a data frame of similar structure as the input but without entries less than a metre resolution.
        """
        depth_from_rounded =df[self.depth_from_col].apply(func)
        depth_to_rounded =df[self.depth_to_col].apply(func)
        df_1 = df.copy(deep=True)
        df_1[self.depth_from_col] = depth_from_rounded
        df_1[self.depth_to_col] = depth_to_rounded
        collapsed = (df_1[self.depth_from_col] == df_1[self.depth_to_col])
        if remove_collapsed:
            df_1 = df_1[~collapsed]
        return df_1

    def assess_num_collapsed(self, df, func=np.round):
        """How many records would be removed from the lithology records if rounding from/to depths

        Args:
            df (pandas data frame): bore lithology data
            func (callable): rounding function callable with a signature similar to `np.round`
        Returns:
            (int): the number of entries that would be collapsed from/to depths
        """
        tmp = self.round_to_metre_depths(df, func)
        collapsed = (tmp[self.depth_from_col] == tmp[self.depth_to_col])
        return collapsed.sum()

# Remove if indeed redundant/superseded
# def slice_above(lithology_df, lower_bound_raster, drop_na=True):
#     df = lithology_df.copy(deep=True)
#     data_grid = lower_bound_raster.read(1)
#     lower_bound_values = raster_drill_df(df, lower_bound_raster, data_grid)
#     if drop_na:
#         df = df[pd.notna(lower_bound_values)]
#         lower_bound_values = lower_bound_values[pd.notna(lower_bound_values)]
#     df_slice=df.loc[(df[DEPTH_FROM_AHD_COL] >= lower_bound_values) & (df[DEPTH_TO_AHD_COL] >= lower_bound_values)]
#     return df_slice

def z_index_for_ahd_functor(a=1, b=50):
    """Creates a linear univariate function for height to index conversions

    Args:
        a (float): slope - defaults to one
        b (float): height offset

    Returns:
        (callable): univariate linear function
    """
    def z_index_for_ahd(ahd):
        return a * ahd + b
    return z_index_for_ahd

def average_slices(slices):
    """Gets the average values over numeric slices

    Args:
        slices (list of 2D np arrays): slices to average

    """

    # TODO there are ways to make this more efficient, e.g. if we get a volume instead of a list of slices
    # my_volume
    # my_sub_volume = my_volume[:,:,from:to:by] or something like that
    # the_average = np.average(my_sub_volume, axis=2).shape
    # for now:
    if len(slices) < 1:
        raise ZeroDivisionError("There must be at least one slice to average over")
    summed = np.empty(slices[0].shape)
    summed = 0.0
    for i in range(len(slices)):
        summed = summed + slices[i]
    return summed / len(slices)


def burn_volume_func(func_below, func_above, volume, surface_raster, height_to_z, below=False, ignore_nan=False, inclusive=False):
    """
    Reusable function, not for end user. Process parts of a xyz volume given a surface, below or above the intersection of the volume with the surface
    """
    dim_x,dim_y,dim_z=volume.shape
    z_index_max = dim_z-1
    # TODO if surface_raster.shape[0] != dim_x or surface_raster.shape[1] != dim_y
    for x in np.arange(0,dim_x,1):
        for y in np.arange(0,dim_y,1):
            # From the original code I had retrieved something I cannot understand (why 30??)
            # erode_until=-(surface_raster.astype(int)-30)[x,y]
            dem_height = surface_raster[x,y]
            if np.isnan(dem_height):
                if not ignore_nan:
                    volume[x,y,:]=np.nan
            else:
                z_height = height_to_z(dem_height)
                z_height = min(z_index_max, max(0.0, z_height))
                z_height = int(round(z_height))
                zh_nan = z_height
                if below:
                    if inclusive:
                        zh_nan = zh_nan + 1
                        zh_nan = min(z_index_max, max(0.0, zh_nan))
                    func_below(volume, x, y, zh_nan)
                else:
                    if not inclusive:
                        zh_nan = zh_nan + 1
                        zh_nan = min(z_index_max, max(0.0, zh_nan))
                    func_above(volume, x, y, zh_nan)

def volume_value_at(volume, slice_surface, height_to_z, x, y):
    """read a volume value at an x/y/z location.

    Args:
        volume (3D ndarray): volume to drill
        slice_surface (2D numpy array): array of elevations
        height_to_z (callable): converter from datum height to z index
        x (float): easting index
        y (float): northing index

    Returns:
        (float): the value in the volume at the specified point
    """
    dim_z=volume.shape[2]
    z_index_max = dim_z-1
    slice_height = slice_surface[x,y]
    def to_int(x):  # may be custom later
        return int(np.floor(x))
    if np.isnan(slice_height):
        return np.nan
    else:
        z_height = to_int(height_to_z(slice_height))
        if z_height < 0:
            return np.nan
        elif z_height > z_index_max:
            return np.nan
        else:
            z = z_height
            return volume[x,y,z]

def slice_volume(volume, slice_surface, height_to_z):
    """read values in a volume along a slicing surface

    Args:
        volume (3D ndarray): volume to drill
        slice_surface (2D numpy array): array of elevations
        height_to_z (callable): converter from datum height to z index

    Returns:
        (float): the value in the volume at the specified point
    """
    dim_x,dim_y,_=volume.shape
    # TODO if surface_raster.shape[0] != dim_x or surface_raster.shape[1] != dim_y
    result = np.empty((dim_x,dim_y))
    for x in np.arange(0,dim_x,1):
        for y in np.arange(0,dim_y,1):
            result[x,y] = volume_value_at(volume, slice_surface, height_to_z, x, y)
    return result

class SliceOperation:
    """Helper class to perform slicing operations on a 3D volume.

    Attributes:
        dem_array_zeroes_infill (2D array): An array, DEM for the grid at the same x/y resolution as the volume to be sliced.
        z_index_for_ahd (callable): bijection from a z index in the volume to its AHD height
    """
    def __init__(self, dem_array_zeroes_infill, z_index_for_ahd):
        """initialize a slice operator for a given grid size

        Args:
            dem_array_zeroes_infill (2D array): An array, DEM for the grid at the same x/y resolution as the volume to be sliced.
            z_index_for_ahd (callable): bijection from a z index in the volume to its AHD height
        """
        self.dem_array_zeroes_infill = dem_array_zeroes_infill
        self.z_index_for_ahd = z_index_for_ahd

    @staticmethod
    def arithmetic_average(slices):
        k_average = np.empty(slices[0].shape)
        k_average = 0.0
        for i in range(len(slices)):
            k_average = k_average + slices[i]
        k_average = k_average / len(slices)
        return k_average

    @staticmethod
    def arithmetic_average_int(slices):
        return np.round(SliceOperation.arithmetic_average(slices))

    def reduce_slices_at_depths(self, volume, from_depth, to_depth, reduce_func):
        """Slice a volume at every meter between two depths below ground level, and reduce the resulting 'vertical' values to a single statistic.

        Args:
            volume (3D array): volume of interest
            from_depth (numeric): top level depth below ground
            to_depth (numeric): bottom level depth below ground. `from_depth` is less than `to_depth`
            reduce_func (callable): A function that takes in a list of 2D grids and returns one 2D grid

        Returns:
            (2D Array): reduced values (e.g. averaged) for each grid geolocation between the two depths below ground.
        """
        slices = [slice_volume(volume, self.dem_array_zeroes_infill - depth, self.z_index_for_ahd) for depth in range(from_depth, to_depth+1)]
        return reduce_func(slices)

    def from_ahd_to_depth_below_ground_level(self, volume, from_depth, to_depth):
        """Slice a volume at every meter between two depths below ground level

        Args:
            volume (3D array): volume of interest
            from_depth (numeric): top level depth below ground
            to_depth (numeric): bottom level depth below ground. `from_depth` is less than `to_depth`
        Returns:
            (3D Array): volume values between the two depths below ground.
        """
        # Note: may not be the most computationally efficient, but deal later.
        depths = range(from_depth, to_depth+1)
        slices = [slice_volume(volume, self.dem_array_zeroes_infill - depth, self.z_index_for_ahd) for depth in depths]
        nx, ny, _ = volume.shape
        nz = len(depths)
        result = np.empty([nx, ny, nz])
        for i in range(nz):
            ii = (nz-1) - i
            result[:,:,i] = slices[ii]
        return result


def burn_volume(volume, surface_raster, height_to_z, below=False, ignore_nan=False, inclusive=False):
    """
    Burn out parts of a xyz volume given a surface, below or above the intersection of the volume with the surface

    :volume: volume to modify
    :type: 3D numpy

    :surface_raster: AHD coordinate at which to slice the data frame for lithology observations
    :type: 2D numpy

    :height_to_z: Number of neighbors to pass to KNeighborsClassifier
    :type: a function to convert the surface raster value (height for a DEM) to a corresponding z-index in the volume.

    :below: should the part below or above be burnt out
    :type: bool

    :ignore_nan: If the surface to burn from has NaNs, should it mask out the whole corresponding cells in the volume
    :type: bool

    :inclusive: is the cell in the volume cut by the surface included in the burning out (i.e. set to NaN) or its value kept?
    :type: bool
    """
    def nan_below_z(volume, x, y, z):
        volume[x, y,0:z]=np.nan

    def nan_above_z(volume, x, y, z):
        volume[x, y,z:]=np.nan

    burn_volume_func(nan_below_z, nan_above_z, volume, surface_raster, height_to_z, below, ignore_nan, inclusive)


def set_at_surface_boundary(volume, surface_raster, height_to_z, value=0.0, ignore_nan=False):
    """
    Burn out parts of a xyz volume given a surface, below or above the intersection of the volume with the surface

    :volume: volume to modify
    :type: 3D numpy

    :surface_raster: AHD coordinate at which to slice the data frame for lithology observations
    :type: 2D numpy

    :height_to_z: Number of neighbors to pass to KNeighborsClassifier
    :type: a function to convert the surface raster value (height for a DEM) to a corresponding z-index in the volume.

    :below: should the part below or above be burnt out
    :type: bool

    :ignore_nan: If the surface to burn from has NaNs, should it mask out the whole corresponding cells in the volume
    :type: bool

    :inclusive: is the cell in the volume cut by the surface included in the burning out (i.e. set to NaN) or its value kept?
    :type: bool
    """
    def set_at_z(volume, x, y, z):
        volume[x, y,z]=value

    burn_volume_func(set_at_z, set_at_z, volume, surface_raster, height_to_z, below=False, ignore_nan=ignore_nan, inclusive=False)


def get_bbox(geo_pd):
    """Gets the bounding box of a geopandas dataframe

    Args:
        geo_pd (geopandas): shape from which we can get the bounding box as a basis for the extend of the meshgrid

    Return:
        (tuple of floats): bbox (x_min, y_min, x_max, y_max)
    """
    return (geo_pd.total_bounds[0], geo_pd.total_bounds[1], geo_pd.total_bounds[2], geo_pd.total_bounds[3])

def cookie_cut_gpd(gpd_df, x_min, x_max, y_min, y_max, fractions=None):
    """Cookie cut a geopandas data frame, typically bore locations.

    Args:
        gpd_df (geopandas): geopandas data
        x_min (numeric): lower x coordinate
        x_max (numeric): upper x coordinate
        y_min (numeric): lower y coordinate
        y_max (numeric): upper y coordinate
        fractions (list of floats): lists of fractions from the original (x_min, x_max, y_min, y_max) box at which to clip the data. Defaults to the centered third of original width/height (i.e. 1/9th of the area)
    Return:
        (TBC): a geopandas data frame (or view thereof)
    """
    if fractions is None:
        fractions = [0.33, 0.67, 0.33, 0.67]
    xmin = x_min + fractions[0] * (x_max-x_min)
    xmax = x_min + fractions[1] * (x_max-x_min)
    ymin = y_min + fractions[2] * (y_max-y_min)
    ymax = y_min + fractions[3] * (y_max-y_min)
    return gpd_df.cx[xmin:xmax, ymin:ymax]


def create_meshgrid_cartesian(x_min, x_max, y_min, y_max, grid_res):
    """Create a 2D meshgrid to be used with numpy for vectorized operations and Mayavi visualisation.

    Args:
        x_min (numeric): lower x coordinate
        x_max (numeric): upper x coordinate
        y_min (numeric): lower y coordinate
        y_max (numeric): upper y coordinate
        grid_res (numeric): x and y resolution of the grid we create

    Return:
        (list of 2 2dim numpy.ndarray): 2-D coordinate arrays for vectorized evaluations of 2-D scalar/vector fields.
            The arrays are ordered such that the first dimension relates to the X coordinate and the second the Y coordinate. This is done such that
            the 2D coordinate arrays work as-is with visualisation with Mayavi without unnecessary transpose operations.
    """
    # The use of indexing='ij' deserves an explanation, as it is counter intuitive. The nupmy doc states
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    # In the 2D case with inputs of length M and N, the outputs are of shame N, M for 'xy' indexing and M, N for 'ij' indexing
    # We want an output that preserves the order of x then y coordinates, so we have to use indexing='ij'  instead of indexing='xy' otherwise the dim order is swapped, and
    # later on for mayavi visualizations we need to swap them back, which leads to confusions.
    return np.meshgrid(np.arange(x_min, x_max, grid_res),np.arange(y_min, y_max, grid_res), indexing='ij')

def create_meshgrid(geo_pd, grid_res):
    """Create a 2D meshgrid to be used with numpy for vectorized operations and Mayavi visualisation.

    Args:
        geo_pd (geopandas): shape from which we can get the bounding box as a basis for the extend of the meshgrid
        grid_res (numeric): x and y resolution of the grid we create

    Return:
        (list of 2 2dim numpy.ndarray): 2-D coordinate arrays for vectorized evaluations of 2-D scalar/vector fields.
            The arrays are ordered such that the first dimension relates to the X coordinate and the second the Y coordinate. This is done such that
            the 2D coordinate arrays work as-is with visualisation with Mayavi without unnecessary transpose operations.
    """
    x_min, y_min, x_max, y_max = get_bbox(geo_pd)
    return create_meshgrid_cartesian(x_min, x_max, y_min, y_max, grid_res)

def intersecting_bounds(bounding_boxes):
    """Gets the bounding box coordinates common to several spatial data sets

        Args:
            bounding_boxes (list): list of bounding boxes
        Returns:
            (tuple): x_min,x_max,y_min,y_max

        Example:
            >>> x_min,x_max,y_min,y_max  = intersecting_bounds( get_bbox(bore_locations), dem.bounds )
    """
    n_bb = len(bounding_boxes)
    # Given the order of the values in the bounding boxes, to get xmin,xmax,ymin,ymax we reorder
    x = [[bounding_boxes[i][j] for i in range(n_bb)] for j in [0, 2, 1, 3]]
    x_min = max(x[0])
    x_max = min(x[1])
    y_min = max(x[2])
    y_max = min(x[3])
    return x_min,x_max,y_min,y_max


def vstacked_points(xx, yy):
    g = (xx, yy)
    m = [np.ravel(pt) for pt in g]
    points = np.vstack(m)
    return points

def surface_array(raster, x_min, y_min, x_max, y_max, grid_res):
    """Creates an isometric 2D grid from a raster, effectively resampling.

    This resampling is done to facilitate downstream operations on gridded interpolated volumes.

    Args:
    raster (rasterio dataset): raster to sample from, typically DEM elevations
        x_min (numeric): lower x coordinate
        x_max (numeric): upper x coordinate
        y_min (numeric): lower y coordinate
        y_max (numeric): upper y coordinate
        grid_res (numeric): x and y resolution of the grid we create

    Return:
        (2dim numpy.ndarray): grid of raster values, typically DEM elevations
    """
    xx, yy = create_meshgrid_cartesian(x_min, x_max, y_min, y_max, grid_res)
    points = vstacked_points(xx, yy)
    num_points=points.shape[1]
    band_1 = raster.read(1)
    z = []
    nd = np.float32(raster.nodata)
    for point in np.arange(0,num_points):
        x=points[0,point]
        y=points[1,point]
        nrow, ncol = band_1.shape
        # return band_1.shape
        row, col = raster.index(x,y)
        # July 2018: Change of behavior with (package versions??). At runtime.
        # Python dynamic typing SNAFU...
        row = int(row)
        col = int(col)
        if (row < nrow and col < ncol and row >= 0 and col >= 0):
            result=band_1[row,col]
            if (result == nd):
                result = np.nan
        else:
            result = np.nan
        z=np.append(z,result)
    # z=z.clip(0) This was probably for the DEM assuming all is above sea level?
    #return (z.shape, xx.shape, num_points)
    dem_array=z.reshape(xx.shape)
    return dem_array



# class_value = 3.0
# color_name = 'yellow'
# single_litho = extract_single_lithology_class_3d(test, class_value)
# mlab.figure(size=(800, 800))
# # s = np.flip(np.flip(test,axis=2), axis=0)
# s = flip(flip(single_litho,axis=2), axis=0)
# vol = mlab.contour3d(s, contours=[class_value-0.5], color=to_rgb(color_name))
# dem_surf = mlab.surf(xx.T, yy.T, np.flipud(dem_array), warp_scale=10, colormap='terrain')
# mlab.ylabel(EASTING_COL)
# mlab.xlabel(NORTHING_COL)
# mlab.zlabel('mAHD')

# mlab.outline()
# mlab.show()

def pad_training_set_functor(classes):
    """Create a function that pads a training set so that all possible classes are present"""
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



class GeospatialDataFrameColumnNames(object):
    """Operations on data frames with 3D spatial information of lithology logs.
    The purpose of this class is to adapt 'pyela' operations
    to different data without requiring renaming columns.

    Attributes:
        easting_col (str): name of the data frame column for easting
        northing_col (str): name of the data frame column for northing
        depth_from_ahd_col (str): name of the data frame column for the height of the top of the soil column (ahd stands for for australian height datum, but not restricted)
        depth_to_ahd_col (str): name of the data frame column for the height of the bottom of the soil column (ahd stands for for australian height datum, but not restricted)
    """

    def __init__(self, easting_col=EASTING_COL, northing_col=NORTHING_COL, depth_from_ahd_col=DEPTH_FROM_AHD_COL, depth_to_ahd_col=DEPTH_TO_AHD_COL):
        """Constructor, operations on data frames with 3D spatial information of lithology logs

            Args:
                easting_col (str): name of the data frame column for easting
                northing_col (str): name of the data frame column for northing
                depth_from_ahd_col (str): name of the data frame column for the height of the top of the soil column (ahd stands for for australian height datum, but not restricted)
                depth_to_ahd_col (str): name of the data frame column for the height of the bottom of the soil column (ahd stands for for australian height datum, but not restricted)
        """
        self.easting_col = easting_col
        self.northing_col = northing_col
        self.depth_from_ahd_col = depth_from_ahd_col
        self.depth_to_ahd_col = depth_to_ahd_col

    def lithologydata_slice_depth(self, df, slice_depth):
        """
        Subset data frame with entries at a specified AHD coordinate

            Args:
                df (pandas data frame): bore lithology data
                slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations

            Returns:
                a (view of a) data frame, a subset of the input data frame,
                entries intersecting with the specified slice depth
        """
        df_slice=df.loc[(df[self.depth_from_ahd_col] >= slice_depth) & (df[self.depth_to_ahd_col] <= slice_depth)]
        return df_slice

    # The following was spurred by trying to get more data in KNN cross-validation, but this may be a dubious method to increase the data pool. Park.
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

    def get_lithology_observations_for_depth(self, df, slice_depth, column_name ):
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
        df_slice = self.lithologydata_slice_depth(df, slice_depth)
        df_1 = df_slice[np.isnan(df_slice[column_name]) == False]
        return df_1

    def extract_bore_class_num(self, bore_log_df, column_name):
        """Gets the columns easting, northing, primary lithology class number, AHD depth 'from' and 'to' from a bore data log

            Args:
                bore_log_df (pandas data frame): bore lithology data
                column_name (str): name of the column of interest e.g. lithology descriptions or classes
        """
        xx = bore_log_df[self.easting_col].values
        yy = bore_log_df[self.northing_col].values
        ss = bore_log_df[column_name].values
        zz_from = bore_log_df[self.depth_from_ahd_col].values
        zz_to = bore_log_df[self.depth_to_ahd_col].values
        return xx, yy, zz_from, zz_to, ss

    def make_training_set(self, observations, column_name):
        """Create a training set suitable for machine learning by e.g. scikit-learn out of a georeferenced data frame.

            Args:
                observations (pandas data frame): bore lithology data
                column_name (str): name of the column of interest e.g. lithology descriptions or classes
        """
        # X = observations.as_matrix(columns=[EASTING_COL, NORTHING_COL])
        X = observations[[self.easting_col, self.northing_col]].values
        y = np.array(observations[column_name])
        #NOTE: should I also do e.g.:
        #shuffle_index = np.random.permutation(len(y))
        #X, y = X[shuffle_index], y[shuffle_index]
        return (X, y)

    def get_knn_model(self, df, column_name, slice_depth, n_neighbours):
        """Train a K-nearest neighbours model for a given plane

        Args:
            df (data frame):
            column_name (str):
            slice_depth (numeric):
            n_neighbours (int):

        Returns:
            KNeighborsClassifier: trained classifier.
        """
        df_1 = self.get_lithology_observations_for_depth(df, slice_depth, column_name)
        X, y = self.make_training_set(df_1, column_name)
        if n_neighbours > len(df_1):
            return None
        else:
            knn = neighbors.KNeighborsClassifier(n_neighbours, weights = KNN_WEIGHTING).fit(X, y)
            return knn

    def class_probability_estimates_depth(self, df, column_name, slice_depth, n_neighbours, mesh_grid, func_training_set=None):
        """Subset data frame with entries at a specified AHD coordinate

            Args:
                df (pandas data frame): bore lithology data
                column_name (str): name of the column with string information to use to strip entries with missing lithology information
                slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations
                n_neighbours (int): number of nearest neighbours
                mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)
                func_training_set (callable):  a function to processing the training set (e.g. completing dummy with dummy classes, other not present in the trainining set)

            Returns:
                a list of numpy arrays, shaped like the meshgrid.
        """
        df_1 = self.get_lithology_observations_for_depth(df, slice_depth, column_name)
        X, y = self.make_training_set(df_1, column_name)
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

    def class_probability_estimates_depth_bbox(self, df, column_name, slice_depth, n_neighbours, geo_pd, grid_res = 100, func_training_set=None):
        """Interpolate over a volume the probability of each lithology class

            Args:
                df (pandas data frame): bore lithology data, spatially georeferenced
                column_name (str): name of the column with numeric codes for lithology classes
                z_ahd_coords (iterable of int): datum heights at which to interpolate. Must be equal to the length of the last dimension of the volume
                n_neighbours (int): number of nearest neighbours
                mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)

            Returns:
                numpy array, predicted values over the grid.
        """
        mesh_grid = create_meshgrid(geo_pd, grid_res)
        return self.class_probability_estimates_depth(df, column_name, slice_depth, n_neighbours, mesh_grid, func_training_set)

    def get_lithology_classes_probabilities(self, lithologies, shape, df, column_name, z_ahd_coords, n_neighbours, mesh_grid):
        """Interpolate over a volume the probability of each lithology class

            Args:
                shape (tuple of ints): shape of the output volumes to interpolate over
                df (pandas data frame): bore lithology data, spatially georeferenced
                column_name (str): name of the column with numeric codes for lithology classes
                z_ahd_coords (iterable of int): datum heights at which to interpolate. Must be equal to the length of the last dimension of the volume
                n_neighbours (int): number of nearest neighbours
                mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)

            Returns:
                numpy array, predicted values over the grid.
        """
        dim_x,dim_y,dim_z = shape
        vol_template=np.empty((dim_x,dim_y,dim_z))
        classprob_3d_arrays=[vol_template.copy() for i in lithologies]
        n_classes = len(lithologies)
        pad_training_set = pad_training_set_functor(lithologies)
        # iterate over all slices
        for z_index,ahd_height in enumerate(z_ahd_coords):
            result=self.class_probability_estimates_depth(df, column_name, ahd_height, n_neighbours, mesh_grid, func_training_set = pad_training_set)
            for i in range(n_classes):
                classprob_3d_arrays[i][:,:,z_index]=result[i]
        return classprob_3d_arrays

    def interpolate_lithologydata_slice_depth(self, df, column_name, slice_depth, n_neighbours, mesh_grid):
        """Interpolate lithology data

            Args:
                df (pandas data frame): bore lithology data
                slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations
                n_neighbours (int): number of nearest neighbours
                mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)

            Returns:
                numpy array, predicted values over the grid.
        """
        knn = self.get_knn_model(df, column_name, slice_depth, n_neighbours)
        return interpolate_over_meshgrid(knn, mesh_grid)

    def interpolate_lithologydata_slice_depth_bbox(self, df, column_name, slice_depth, n_neighbours, geo_pd, grid_res = 100):
        """Interpolate lithology data

            Args:
                df (pandas data frame): bore lithology data
                slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations
                n_neighbours (int): number of nearest neighbours
                geo_pd (geopandas df): vector of spatial data from which to get the bounds of interest (bounding box)
                grid_res (int): grid resolution in m for x and y.

            Returns:
                numpy array, predicted values over the grid.
        """
        mesh_grid = create_meshgrid(geo_pd, grid_res)
        return self.interpolate_lithologydata_slice_depth(df, column_name, slice_depth, n_neighbours, mesh_grid)


def interpolate_over_meshgrid(predicting_algorithm, mesh_grid):
    """Interpolate lithology data

        Args:
            predicting_algorithm (algorithm with a predict method.): trained algorithm such as the K Nearest Neighbours in scikit (KNN)
            mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)

        Returns:
            numpy array, predicted values over the grid.
    """
    xx, yy = mesh_grid
    if predicting_algorithm is None:
        # the training set was too small and prediction cannot be made (odd that scikit would have let us train still)
        predicted = np.full(xx.shape, np.nan)
    else:
        predicted = predicting_algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
        predicted = predicted.reshape(xx.shape)
    return predicted

class DepthCoverage:
    """Helper class to round lithology record classes to the nearest metre of depth

    Attributes:
        df (data frame):
        depth_from_col (str): Column name  storing "from depth" information
        depth_to_col (str): Column name  storing "to depth" information
        group_col (str):
    """
    def __init__(self, df, group_col = BORE_ID_COL, depth_from_col=DEPTH_FROM_AHD_COL, depth_to_col=DEPTH_TO_AHD_COL):
        """Helper class to round lithology record classes to the nearest metre of depth

        Args:
            group_col (str):
            depth_from_col (str): Column name  storing "from depth" information
            depth_to_col (str): Column name  storing "to depth" information
        """
        self.depth_from_col = depth_from_col
        self.depth_to_col = depth_to_col
        self.group_col = group_col
        x=df[np.isnan(df[depth_from_col]) == False]
        df_by_boreid = x.groupby(group_col)
        tops = df_by_boreid[depth_from_col].max()
        bottoms = df_by_boreid[depth_to_col].min()
        self.bore_ahd_depths = pd.concat([tops, bottoms], axis=1).reset_index()
    def bore_at_height(self, datum_height):
        return np.sum( np.logical_and((self.bore_ahd_depths[self.depth_from_col].values >= datum_height), (self.bore_ahd_depths[self.depth_to_col].values < datum_height)) )
    def highest_top(self):
        return self.bore_ahd_depths[self.depth_from_col].max()
    def lowest_bottom(self):
        return self.bore_ahd_depths[self.depth_to_col].min()
    def get_counts(self):
        r = np.array([ float(i) for i in range( int(np.floor(self.lowest_bottom())), int(np.floor(self.highest_top())) ) ])
        cc = np.array([ self.bore_at_height(i) for i in r ])
        return (r, cc)
    def get_range(self, cutoff):
        r, cc = self.get_counts()
        w = np.argwhere(cc > cutoff)
        lowest_ind = w[0][0]
        highest_ind = w[-1][0]
        return r[lowest_ind], r[highest_ind]


class GridInterpolation:
    """Operations interpolating over a grid using a trained model
    The purpose of this class is to adapt 'pyela' operations
    to different data without requiring renaming columns.

    Attributes:
        dfcn (GeospatialDataFrameColumnNames):
        northing_col (str): name of the data frame column for northing
        depth_from_ahd_col (str): name of the data frame column for the height of the top of the soil column (ahd stands for for australian height datum, but not restricted)
        depth_to_ahd_col (str): name of the data frame column for the height of the bottom of the soil column (ahd stands for for australian height datum, but not restricted)
    """

    def __init__(self, easting_col=EASTING_COL, northing_col=NORTHING_COL, depth_from_ahd_col=DEPTH_FROM_AHD_COL, depth_to_ahd_col=DEPTH_TO_AHD_COL):
        """Constructor, operations interpolating over a grid using a trained model

            Args:
                easting_col (str): name of the data frame column for easting
                northing_col (str): name of the data frame column for northing
                depth_from_ahd_col (str): name of the data frame column for the height of the top of the soil column (ahd stands for for australian height datum, but not restricted)
                depth_to_ahd_col (str): name of the data frame column for the height of the bottom of the soil column (ahd stands for for australian height datum, but not restricted)
        """
        self.dfcn = GeospatialDataFrameColumnNames(easting_col, northing_col, depth_from_ahd_col, depth_to_ahd_col)


    def interpolate_volume(self, volume, df, column_name, z_ahd_coords, n_neighbours, mesh_grid):
        """Interpolate lithology data over a volume

            Args:
                volume (ndarray): 3d volume to fill with interpolated lithologies
                df (pandas data frame): bore lithology data
                column_name (str): name of the column with numeric codes for lithology classes
                z_ahd_coords (iterable of int): datum heights at which to interpolate. Must be equal to the length of the last dimension of the volume
                n_neighbours (int): number of nearest neighbours
                mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)

            Returns:
                numpy array, predicted values over the grid.
        """
        dim_x,dim_y = mesh_grid[0].shape
        dim_z = len(z_ahd_coords)
        if volume.shape[0] != dim_x or volume.shape[1] != dim_y or volume.shape[2] != dim_z:
            raise Error("Incompatible dimensions in arguments")
        for index,ahd_height in enumerate(z_ahd_coords):
            surface = self.interpolate_lithologydata_slice_depth(df, column_name, ahd_height, n_neighbours, mesh_grid)
            volume[:,:,index]=surface

    def interpolate_lithologydata_slice_depth(self, df, column_name, slice_depth, n_neighbours, mesh_grid):
        """Interpolate lithology data

            Args:
                df (pandas data frame): bore lithology data
                column_name (str): name of the column with string information to use to strip entries with missing lithology information
                slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations
                n_neighbours (int): number of nearest neighbours
                mesh_grid (tuple): coordinate matrices to interpolate over (numpy.meshgrid)

            Returns:
                numpy array, predicted values over the grid.
        """
        knn = self.get_knn_model(df, column_name, slice_depth, n_neighbours)
        return interpolate_over_meshgrid(knn, mesh_grid)

    def get_knn_model(self, df, column_name, slice_depth, n_neighbours):
        """Train a K-nearest neighbours model for a given plane

            Args:
                df (pandas data frame): bore lithology data
                column_name (str): name of the column with string information to use to strip entries with missing lithology information
                slice_depth (float): AHD coordinate at which to slice the data frame for lithology observations
                n_neighbours (int): number of nearest neighbours

        Returns:
            KNeighborsClassifier: trained classifier.
        """
        df_1 = self.get_lithology_observations_for_depth(df, slice_depth, column_name)
        X, y = self.make_training_set(df_1, column_name)
        if n_neighbours > len(df_1):
            return None
        else:
            knn = neighbors.KNeighborsClassifier(n_neighbours, weights = KNN_WEIGHTING).fit(X, y)
            return knn

    def get_lithology_observations_for_depth(self, df, slice_depth, column_name ):
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
        df_slice=self.dfcn.lithologydata_slice_depth(df, slice_depth)
        df_1=df_slice[np.isnan(df_slice[column_name]) == False]
        return df_1

    def make_training_set(self, observations, column_name):
        """Create a training set from a set of geolocated observations

            Args:
                observations (pandas data frame): bore lithology data with geocoordinates
                column_name (str): name of the column with string information to use to strip entries with missing lithology information

            Returns:
                (tuple): observations and predictors (geolocation).
        """
        X = observations[[self.dfcn.easting_col, self.dfcn.northing_col]].values
        y = np.array(observations[column_name])
        #NOTE: should I also do e.g.:
        #shuffle_index = np.random.permutation(len(y))
        #X, y = X[shuffle_index], y[shuffle_index]
        return (X, y)

