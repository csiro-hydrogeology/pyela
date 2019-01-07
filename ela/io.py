import numpy as np
import rasterio
from ela.visual import to_color_image, get_color_component, to_carto


# TOCHECK: appears unused. What was the intent?
# # note on generic slicing: something like the following. For now assume some things in the rgba mapping to image interms of dims
# # https://stackoverflow.com/a/37729566/2752565
# def simple_slice(arr, inds, axis):
#     # this does the same as np.take() except only supports simple slicing, not
#     # advanced indexing, and thus is much faster
#     sl = [slice(None)] * arr.ndim
#     sl[axis] = inds
#     return arr[sl]

class GeotiffExporter:
    """Helper class to save matrices into georeferenced, GeoTiff images

    Attributes:
        crs (str, dict, or CRS): The coordinate reference system.
        transform (Affine instance): Affine transformation mapping the pixel space to geographic space.
    """
    def __init__(self, crs, transform):
        """initialize this with a coordinate reference system object and an affine transform. See rasterio.
        
        Args:
            crs (str, dict, or CRS): The coordinate reference system.
            transform (Affine instance): Affine transformation mapping the pixel space to geographic space.
        """
        self.crs = crs
        self.transform = transform

    def export_rgb_geotiff(self, matrix, full_filename, classes_cmap):
        """Save a matrix of numeric classes to an image, using a color to convert numeric values to colors

        Args:
            matrix (ndarray): numpy array, 2 dims
            full_filename (str): Full file name to save the GeoTiff image to.
            classes_cmap (dict): color map with keys as zero based numeric integers and values RGBA tuples.

        """
        n_bands = 3
        x_dataset = rasterio.open(full_filename, 'w', driver='GTiff',
                                height=matrix.shape[0], width=matrix.shape[1],
                                count=n_bands, dtype= 'uint8',
                                crs=self.crs, transform=self.transform)
        colors_array = to_color_image(matrix, classes_cmap)
        r = get_color_component(colors_array, 0)
        g = get_color_component(colors_array, 1)
        b = get_color_component(colors_array, 2)
        x_dataset.write(r, 1)
        x_dataset.write(g, 2)
        x_dataset.write(b, 3)
        x_dataset.close()

    def export_geotiff(self, matrix, full_filename, classes_cmap):
        """Save a matrix of numeric classes to an image, using a color to convert numeric values to colors

        Args:
            matrix (ndarray): numpy array, 2 dims
            full_filename (str): Full file name to save the GeoTiff image to.
            classes_cmap (dict): color map with keys as zero based numeric integers and values RGBA tuples.

        """
        x_dataset = rasterio.open(full_filename, 'w', driver='GTiff',
                                height=matrix.shape[0], width=matrix.shape[1],
                                count=1, dtype= 'float64',
                                crs=self.crs, transform=self.transform)
        x_dataset.write(matrix, 1)
        # Tried to write a colormap by inference from https://github.com/mapbox/rasterio/blob/master/tests/test_colormap.py but seems not to improve things. 
        # the api doc is woefully insufficient: https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.BufferedDatasetWriter.write_colormap
        # x_dataset.write_colormap(1, classes_cmap)
        x_dataset.close()


# TODO: consider if refactor the following, using also ela.spatial.SliceOperation
# def export_avg_at_depth(classes, outdir, fname, from_depth, to_depth, class_cmap):
#     slices = [slice_volume(classes, dem_array_zeroes_infill - depth, z_index_for_ahd) for depth in range(from_depth, to_depth+1)]
#     k_average = np.empty(slices[0].shape)
#     k_average = 0.0
#     for i in range(len(slices)):
#         k_average = k_average + slices[i]
#     k_average = k_average / len(slices)
#     x = to_carto(k_average)
#     ge.export_geotiff(x, os.path.join(outdir, fname), class_cmap)

# def export_avg_rgb_at_depth(classes, outdir, fname, from_depth, to_depth, class_cmap):
#     slices = [slice_volume(classes, dem_array_zeroes_infill - depth, z_index_for_ahd) for depth in range(from_depth, to_depth+1)]
#     k_average = np.empty(slices[0].shape)
#     k_average = 0.0
#     for i in range(len(slices)):
#         k_average = k_average + slices[i]
#     k_average = k_average / len(slices)
#     x = to_carto(k_average)
#     ge.export_rgb_geotiff(x, os.path.join(outdir, fname), class_cmap)

