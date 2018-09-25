from matplotlib import colors
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.io.img_tiles as cimgt

from sklearn import neighbors
import numpy as np

from sys import float_info

from ela.textproc import *
from ela.classification import KNN_WEIGHTING

DEFAULT_LITHOLOGY_COLORNAMES = ['sandybrown','gold','chocolate','yellow','lightsteelblue','dimgray','darkgoldenrod']
DEFAULT_LITHOLOGY_COLORNAMES_WITH_UNCLASSIFIED = ['black', 'sandybrown','gold','chocolate','yellow','lightsteelblue','dimgray','darkgoldenrod']
#set an intuitive colormap for further plotting


def cartopy_color_settings(color_names, numeric_classes = None):
    """Builds a discrete color settings for use in cartopy functions.

    Args:
        color_names (list of str): color names e.g. ['navyblue','red']
        numeric_classes (list of int): numeric code for classes. Defaults to zero-based integers if None.

    Returns:
        dict: color map with keys 'cmap', 'bounds', 'norm'
    """
    lithology_cmap = colors.ListedColormap(color_names)
    if numeric_classes is None:
        bounds =  [0] + [(x + 1 - float_info.epsilon) for x in range(len(color_names))]
    else:
        raise Error('custom lithology numeric class codes is not yet supported')
    norm = colors.BoundaryNorm(bounds, lithology_cmap.N)
    return {
        'cmap': lithology_cmap,
        'bounds': bounds,
        'norm': norm
    }

def to_carto(x):
    """Reorient an array X-Y to an array suitable for 'natural' orientation with cartopy

    Args:
        x (ndarray): numpy array, 2 dims

    Returns:
        ndarray: 2D array 
    """
    return np.flip(np.swapaxes(x, 0, 1), 0)


def get_color_component(color_array, color_index):
    """Gets one of the color components in an RGBA image specification. Helper function to save geotiff.

    Args:
        color_array (ndarray): numpy array, 3 dims (x, y, and RGBA)
        color_index (int): index, 0 to 3, for which of the RGBA component to retrieve

    Returns:
        ndarray: 2D array with the color component of interes
    """
    # TODO: checks on dimcneionality and dtype.
    return color_array[:,:,color_index]

def discrete_classes_colormap(color_names):
    """Builds a colormap mapping from zero-indexed class codes to rgba colors

    Args:
        color_names (list of str): color names e.g. ['navyblue','red']

    Returns:
        dict: color map with keys as zero based numeric integers and values RGBA tuples.
    """
    c = [to_rgba_255(x) for x in color_names]
    c_rgba = [(c[i][0],c[i][1],c[i][2],c[i][3]) for i in range(len(c))]
    return dict([(i, c_rgba[i]) for i in range(len(c_rgba))])

def interp(x1, x2, fraction):
    return x1 + (x2 - x1) * fraction

def interp_color(t_c1, t_c2, fraction):
    return [int(interp(t_c1[i], t_c2[i], fraction)) for i in range(len(t_c1))]

def interpolate_rgba(cmap, lower_index, fraction, type='linear'):
    """Interpolate between two colors

    Args:
        cmap (dict): color map with keys as zero based numeric integers and values RGBA tuples.
        lower_index (int): lower_index, key to the color to interpolate from (to the next color at lower_index + 1)
        fraction (float): 0 to 1, weight to interpolate between the two colors
        type (str): ignored; future arg for other interpolation schemes.

    Returns:
        tuple: RGBA values.
    """
    lower_rgba = cmap[lower_index]
    higher_rgba = cmap[lower_index + 1]
    return interp_color(lower_rgba, higher_rgba, fraction)

def to_color_image(x, cmap):
    """Convert a 2D array to a color representation (RGBA)

    Args:
        cmap (dict): color map with keys as zero based numeric integers and values RGBA tuples.
        lower_index (int): lower_index, key to the color to interpolate from (to the next color at lower_index + 1)
        fraction (float): 0 to 1, weight to interpolate between the two colors
        type (str): ignored; future arg for other interpolation schemes.

    Returns:
        tuple: RGBA values.
    """
    dims = [d for d in x.shape]
    dims.append(4)
    r = np.empty(dims, dtype='uint8')
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            cval = x[i,j]
            if(np.isnan(cval)):
                r[i,j] = 255
            else:
                cval_floor = int(np.floor(cval))
                if cval_floor == cval:
                    r[i,j] = cmap[cval]
                else:
                    # we need to interpolate between two colors. 
                    f = cval - cval_floor
                    r[i,j] = interpolate_rgba(cmap, cval_floor, f)
    return r

def legend_fig(k_legend_display_info):
    """Plot a legend for color coded classes. Adapted from https://matplotlib.org/examples/color/named_colors.html

    Args:
        k_legend_display_info (list): values are 3-item tuples (rgb_val(not used), k_label, colorname)

    Returns:
        matplotlib Figure: 
    """
    n = len(k_legend_display_info)
    ncols = 1
    nrows = n
    fig, ax = plt.subplots(figsize=(8, 5))
    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols
    for i, legend_info in enumerate(k_legend_display_info):
        rgb_val, k_label, colorname = legend_info
        col = i % ncols
        row = i // ncols
        y = Y - (row * h) - h
    
        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)
    
        ax.text(xi_text, y, k_label, fontsize=(h * 0.8),
            horizontalalignment='left',
            verticalalignment='center')

        #ax.hlines(y + h * 0.1, xi_line, xf_line,
        #          color=rgb_val, linewidth=(h * 0.6))
        ax.hlines(y + h * 0.1, xi_line, xf_line,
              color=colorname, linewidth=(h * 0.6))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)
    return fig



class LithologiesClassesVisual(object):
    def __init__(self, class_names, color_names, missing_value_color_name):
        self.class_names = class_names
        self.color_names = color_names
        self.color_names_with_missing = [missing_value_color_name] + color_names

    def nb_labels(self):
        return len(self.class_names)

    def nb_labels_with_missing(self):
        return self.nb_labels() + 1


def to_rgb(color_name):  # because anaconda2 seems stuck with Matplotlib 1.5 and there is no to_rgb (there is, in ulterior versions!)
    return colors.colorConverter.to_rgb(color_name)

# a couple of functions to create color maps suitable for mayavi.

def to_rgb_255(colorname):
    rgb = to_rgb(colorname)
    return [int(x * 255) for x in rgb]

def to_rgba_255(colorname, alpha = 255):
    rgb = to_rgb_255(colorname)
    return [rgb[0], rgb[1], rgb[2], alpha]


def plot_lithologydata_slice_points_redo(df, 
    slice_depth, extent, data_proj, 
    near_field_extents, geoms, terrain):
    fig,ax=plt.subplots(1,1,figsize=(15,15), subplot_kw={'projection': data_proj,'extent': extent})
    # fig.clear()
    # ax.clear()
    ax.add_image(terrain, 11)
    ax.add_geometries(geoms[0], ccrs.PlateCarree(),facecolor='none',edgecolor='k',zorder=1)
    ax.add_geometries(geoms[1], ccrs.PlateCarree(),facecolor='none',edgecolor='r',zorder=1)
    for val,label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label.set_text(str(val))
        label.set_position((val,0))  
    for val,label in zip(ax.get_yticks(), ax.get_yticklabels()):   
        label.set_text(str(val))
        label.set_position((0,val))  
    plt.tick_params(bottom=True,top=True,left=True,right=True,labelbottom=True,labeltop=False,labelleft=True,labelright=False)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(style='plain')
    ax.grid(False)
    ax.text(0.1, 0.9, u'\u25B2 \nN ',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=25, 
        color='k',
        family='Arial',
        transform=ax.transAxes)
    ax.set_extent(near_field_extents, crs=data_proj)
    # Note that all of the above is independent of slice depth and background that would not need redoing
    # but Matplotlib befuddles (or rather the interplay with ipywidgets)
    df_slice=df.loc[(df[DEPTH_FROM_COL] <= slice_depth) & (df[DEPTH_TO_COL] >= slice_depth)]
    ax.scatter(df_slice.Easting.values,df_slice.Northing.values)
    plt.title('bore log locations at %s m depth'%(slice_depth), fontsize=20, weight='bold')
    # I cannot fathom why this stuff actually plots anything 
    # via ipywidgets or otherwise since it returns nothing.



def create_background(extent, data_proj, 
    near_field_extents, geoms):
    fig,ax=plt.subplots(1,1,figsize=(15,15), subplot_kw={'projection': data_proj,'extent': extent})
    stamen_terrain = cimgt.StamenTerrain()
    ax.add_image(stamen_terrain, 11)
    ax.add_geometries(geoms[0], ccrs.PlateCarree(),facecolor='none',edgecolor='k',zorder=1)
    ax.add_geometries(geoms[1], ccrs.PlateCarree(),facecolor='none',edgecolor='r',zorder=1)
    for val,label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label.set_text(str(val))
        label.set_position((val,0))  
    for val,label in zip(ax.get_yticks(), ax.get_yticklabels()):   
        label.set_text(str(val))
        label.set_position((0,val))  
    plt.tick_params(bottom=True,top=True,left=True,right=True,labelbottom=True,labeltop=False,labelleft=True,labelright=False)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(style='plain')

    ax.grid(False)

    ax.text(0.1, 0.9, u'\u25B2 \nN ',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=25, 
        color='k',
        family='Arial',
        transform=ax.transAxes)

    ax.set_extent(near_field_extents, crs=data_proj)
    scatter_layer = ax.scatter(near_field_extents[0], near_field_extents[2])
    return (fig, scatter_layer)

def plot_lithologydata_slice_points(df, slice_depth, scatter_layer, fig):
    df_slice=df.loc[(df[DEPTH_FROM_COL] <= slice_depth) & (df[DEPTH_TO_COL] >= slice_depth)]
    plt.title('bore log locations at %s m depth'%(slice_depth), fontsize=20, weight='bold')
    e = df_slice.Easting.values
    n = df_slice.Northing.values
    bore_coords = [[e[i], n[i]] for i in range(0, len(e))]
    scatter_layer.set_offsets(bore_coords)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig

def plot_lithologydata_slice_depth(df, slice_depth, n_neighbours, extent, data_proj, near_field_extents, geoms, gw_subareas, cmap_settings):
    df_slice=df.loc[(df[DEPTH_FROM_AHD_COL] >= slice_depth) & (df[DEPTH_TO_AHD_COL] <= slice_depth)]
    fig,ax=plt.subplots(1,1,figsize=(15,15),subplot_kw={'projection': data_proj,'extent': extent})
    stamen_terrain = cimgt.StamenTerrain()
    ax.add_image(stamen_terrain, 11)
    ax.add_geometries(geoms[0], ccrs.PlateCarree(),facecolor='none',edgecolor='k',zorder=1)
    ax.add_geometries(geoms[1], ccrs.PlateCarree(),facecolor='none',edgecolor='r',zorder=1)

    for i, txt in enumerate(df_slice[PRIMARY_LITHO_COL].values):
        plt.annotate(txt,(df_slice.Easting.values[i],df_slice.Northing.values[i]),fontsize=8,clip_on=True)
    
    for val,label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label.set_text(str(val))
        label.set_position((val,0))  
    
    for val,label in zip(ax.get_yticks(), ax.get_yticklabels()):   
        label.set_text(str(val))
        label.set_position((0,val))  
    
    plt.tick_params(bottom=True,top=True,left=True,right=True,labelbottom=True,labeltop=False,labelleft=True,labelright=False)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(style='plain')
    ax.grid(False)
    ax.text(0.1, 0.9, u'\u25B2 \nN ',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=25, 
        color='k',
        family='Arial',
        transform=ax.transAxes)
    plt.title('KNN facies classification at %s m AHD (neighbours=%s)'%(slice_depth,n_neighbours), fontsize=20, weight='bold')

    df_1=df_slice[df_slice.Lithology_1 != ""]
    # X = df_1.as_matrix(columns=[EASTING_COL, NORTHING_COL])
    X = df_1[[EASTING_COL, NORTHING_COL]].values
    y = np.array(df_1[PRIMARY_LITHO_NUM_COL])
    knn = neighbors.KNeighborsClassifier(n_neighbours, weights = KNN_WEIGHTING).fit(X, y)
    grid_res=100
    
    # max/min bounds
    x_min=gw_subareas.total_bounds[0]
    y_min=gw_subareas.total_bounds[1]
    x_max=gw_subareas.total_bounds[2]
    y_max=gw_subareas.total_bounds[3]
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_res),np.arange(y_min, y_max, grid_res))
    predicted = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    predicted = predicted.reshape(xx.shape)
    plt.pcolormesh(xx, yy, predicted, cmap=cmap_settings['cmap'], norm=cmap_settings['norm'], alpha=0.3)
    ax.set_extent(near_field_extents, crs=data_proj)
