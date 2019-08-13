from matplotlib import colors
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from sklearn import neighbors
import numpy as np

from ela.textproc import *
from ela.classification import KNN_WEIGHTING


# These functions were an attempt to have interactive maps with ipywidgets but proved to be a pain. 
# I may revisit later on but these are parked. 

def plot_lithologydata_slice_points_redo(df, 
    slice_depth, extent, data_proj, 
    near_field_extents, geoms, terrain):
    fig,ax=plt.subplots(1,1,figsize=(15,15), subplot_kw={'projection': data_proj, 'extent': extent})
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
    stamen_terrain = cimgt.Stamen('terrain-background')
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
    stamen_terrain = cimgt.Stamen('terrain-background')
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

