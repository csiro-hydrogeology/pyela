from matplotlib import colors
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.io.img_tiles as cimgt

from wordcloud import WordCloud,STOPWORDS

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
    """Visual information to map to rendering lithology classes

    A parent helper class with lithology classes and color scheme information to apply to data in a set of custom visual rendering using Mayavi

    Attributes:
        class_names (str):
        color_names (str):
        color_names_with_missing (str):
    """
    def __init__(self, class_names, color_names, missing_value_color_name):
        """Define class names of interest in visualised data set, and color coding.
        
        Args:
            class_names (list of str): names of the classes
            color_names (list of str): names of the colors for the classes. See matplotlib doc for suitable names: https://matplotlib.org/examples/color/named_colors.html
            missing_value_color_name (str): name of the color for missing values (NaN)
        """
        self.class_names = class_names
        self.color_names = color_names
        self.color_names_with_missing = [missing_value_color_name] + color_names

    def nb_labels(self):
        """Gets the number of classes. Used for e.g. legend's portioning"""
        return len(self.class_names)

    def nb_labels_with_missing(self):
        """Gets the number of classes plus one for the missing value code. Used for e.g. legend's portioning"""
        return self.nb_labels() + 1


def to_rgb(color_name):  # because anaconda2 seems stuck with Matplotlib 1.5 and there is no to_rgb (there is, in ulterior versions!)
    """
    Returns an *RGB* tuple of three floats from 0-1. shortcut to matplotlib, needed bc anaconda2 seems stuck with Matplotlib 1.5 .

    *color_name* can be an *RGB* or *RGBA* sequence or a string in any of
    several forms:

        1) a letter from the set 'rgbcmykw'
        2) a hex color string, like '#00FFFF'
        3) a standard name, like 'aqua'
        4) a string representation of a float, like '0.4',
            indicating gray on a 0-1 scale

    if *arg* is *RGBA*, the *A* will simply be discarded.
    """
    return colors.colorConverter.to_rgb(color_name)

# a couple of functions to create color maps suitable for mayavi.

def to_rgb_255(colorname):
    """Returns an *RGB* tuple of three ints from 0-255"""
    rgb = to_rgb(colorname)
    return [int(x * 255) for x in rgb]

def to_rgba_255(colorname, alpha = 255):
    """Returns an *RGBA* tuple of 4 ints from 0-255"""
    rgb = to_rgb_255(colorname)
    return [rgb[0], rgb[1], rgb[2], alpha]

def show_wordcloud(text, title = None, max_words=200, max_font_size=40, seed=1, scale=3, figsize=(12, 12)):
    """Plot wordclouds from text

        Args:
            text (str or list of str): text to depict
    """
    if text is list:
        text = ' '.join(text)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=max_words,
        max_font_size=max_font_size, 
        scale=scale,
        random_state=seed
    ).generate(text)
    fig = plt.figure(1, figsize=figsize)
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()