import numpy as np

from mayavi import mlab

from ela.classification import extract_single_lithology_class_3d, extract_bore_primary_litho_class_num
from ela.utils import flip
from ela.visual import to_rgba_255, to_rgb, LithologiesClassesVisual
from ela.textproc import EASTING_COL, NORTHING_COL, DEPTH_FROM_AHD_COL, DEPTH_FROM_COL, DEPTH_TO_AHD_COL, DEPTH_TO_COL, PRIMARY_LITHO_NUM_COL, GEOMETRY_COL


def create_colormap_lut(color_names):
    return np.array([np.array(to_rgba_255(c), dtype=np.uint8) for c in color_names])

#@mlab.draw?
def set_custom_colormap(lut, color_names):
    '''
    Reference: http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html 
    '''
    my_lut = create_colormap_lut(color_names)
    lut.number_of_colors = len(my_lut)
    lut.table = my_lut


class LithologiesClassesVisual3d(LithologiesClassesVisual):
    def __init__(self, class_names, color_names, missing_value_color_name):
        super(LithologiesClassesVisual3d, self).__init__(class_names, color_names, missing_value_color_name)

    def set_litho_class_colormap(self, lut):
        set_custom_colormap(lut, self.color_names)

    def set_litho_class_colormap_with_unclassified(self, lut):
        set_custom_colormap(lut, self.color_names_with_missing)

    def create_plane_cut(self, volume, plane_orientation='x_axes', slice_index = 20, colormap=None):
        f = mlab.pipeline.scalar_field(volume)
        if colormap is None:
            cut = mlab.pipeline.image_plane_widget(f, 
                plane_orientation=plane_orientation, slice_index=slice_index)
            self.set_litho_class_colormap(get_colorscale_lut(cut))
        else:
            cut = mlab.pipeline.image_plane_widget(f, 
                plane_orientation=plane_orientation, slice_index=slice_index, colormap=colormap)
        cut.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
        return cut

    @mlab.show
    def render_classes_planar(self, volume):
        mlab.figure(size=(800, 800))
        s = volume
        x_cut=self.create_plane_cut(s, plane_orientation='x_axes')
        y_cut=self.create_plane_cut(s, plane_orientation='y_axes')
        z_cut=self.create_plane_cut(s, plane_orientation='z_axes')
        mlab.outline()
        mlab.xlabel(EASTING_COL)
        mlab.ylabel(NORTHING_COL)
        mlab.zlabel('mAHD')
        mlab.scalarbar(nb_labels=self.nb_labels())

    @mlab.show
    def render_class(self, volume_lithologies, class_value):
        class_index = int(class_value)
        class_name = self.class_names[class_index]
        color_name = self.color_names[class_index]
        single_litho = extract_single_lithology_class_3d(volume_lithologies, class_value)
        mlab.figure(size=(800, 800))
        s = single_litho
        mlab.contour3d(s, contours=[class_value-0.5], color=to_rgb(color_name))
        mlab.outline()
        mlab.xlabel(EASTING_COL)
        mlab.ylabel(NORTHING_COL)
        mlab.zlabel('mAHD')
        mlab.title(class_name)

    @mlab.show
    def render_proba_class(self, prob_volume, title):
        mlab.figure(size=(800, 800))
        # s = np.flip(np.flip(test,axis=2), axis=0)
        s = prob_volume
        colormap='magma'
        x_cut=self.create_plane_cut(s, plane_orientation='x_axes', colormap=colormap)
        y_cut=self.create_plane_cut(s, plane_orientation='y_axes', colormap=colormap)
        z_cut=self.create_plane_cut(s, plane_orientation='z_axes', colormap=colormap)
        mlab.outline()
        mlab.xlabel(EASTING_COL)
        mlab.ylabel(NORTHING_COL)
        mlab.zlabel('mAHD')
        mlab.title(title)
        mlab.scalarbar(nb_labels=11)

def scale_z_bore_pos_points(x, y, z, s, z_scaling):
    zz = z * z_scaling
    return x, y, zz, s

#######################
#### This section was intended to plot bore data as cylinders, but the result was disappointing
#def process_bore_pos(xyzs):
#    x, y, zf, zt, s = xyzs
#    xx = np.repeat(x, 2)
#    yy = np.repeat(y, 2)
#    c = np.vstack((zf,zt+0.01)).reshape((-1,),order='F')
#    zz = c * Z_SCALING
#    ss = np.repeat(s, 2)
#    return xx, yy, zz, ss

#def f(x): 
#    return process_bore_pos(extract_bore_primary_litho_class_num(x))
#sitesgrp = df_1.groupby(WIN_SITE_ID_COL)
#b = sitesgrp.apply(f)
#######################

class LithologiesClassesOverlayVisual3d(LithologiesClassesVisual3d):
    def __init__(self, class_names, color_names, missing_value_color_name, dem_array_data, z_coords, z_scaling, litho_df):
        super(LithologiesClassesOverlayVisual3d, self).__init__(class_names, color_names, missing_value_color_name)
        # 1D georeferenced data (bore primary litho data)
        x, y, z_from, z_to, s = extract_bore_primary_litho_class_num(litho_df)
        self.bore_data = scale_z_bore_pos_points(x, y, z_to, s, z_scaling)
        # 2d data: DEM
        xg, yg = dem_array_data['mesh_xy']
        dem_a = dem_array_data['dem_array']
        dem_a_scaled = dem_a * z_scaling
        self.dem_mesh = (xg, yg, dem_a_scaled)
        self.dim_x,self.dim_y=xg.shape
        # and the 3D data
        self.dim_z=len(z_coords)
        vol=np.empty((self.dim_x,self.dim_y,self.dim_z))
        # Prepare 3D grid meshes for visualizing the interpolated lithography classes
        # Feels bloated but only way I found to get the visual result I wanted.
        self.xxx=vol.copy()
        self.yyy=vol.copy()
        self.zzz=vol.copy()
        xcoord = xg[:,0]
        for xi in np.arange(self.dim_x):
            self.xxx[xi,:,:]= xcoord[xi]
        ycoord = yg[0,:]
        for yi in np.arange(self.dim_y):
            self.yyy[:,yi,:]= ycoord[yi]
        for zi in z_coords:
            self.zzz[:,:,zi]= zi * z_scaling
        self.POINTS_SCALE_FACTOR = 150

    @mlab.show
    def overlay_bore_classes(self, dem_mesh, vol_mesh, bore_data, vol_colorname, z_label='AHD x 50', points_scale_factor=150.0, title=None):
        x_dem, y_dem, z_dem = dem_mesh
        vol_x, vol_y, vol_z, vol_s = vol_mesh
        bore_x, bore_y, bore_z, bore_s = bore_data
        mlab.figure(size=(1200, 800))
        p3d = mlab.points3d(bore_x, bore_y, bore_z, bore_s, colormap='spectral', scale_factor = points_scale_factor, scale_mode='none')
        self.set_litho_class_colormap_with_unclassified(get_colorscale_lut(p3d))
        mlab.outline()
        mlab.xlabel(EASTING_COL)
        mlab.ylabel(NORTHING_COL)
        mlab.zlabel(z_label)
        surface = mlab.surf(x_dem, y_dem, z_dem, colormap='terrain')
        surface.enable_contours = True
        surface.contour.number_of_contours = 20
        iso_surface = mlab.contour3d(vol_x, vol_y, vol_z, vol_s, color=to_rgb(vol_colorname))
        if not title is None:
            mlab.title(title)
        # iso_surface.actor.property.color = (0.0, 1.0, 0.0)
        # iso_surface.actor.property.opacity = 0.0881
        # iso_surface.actor.property.representation = 'wireframe'

    def view_overlay(self, litho_class_value, lithology_3d_array):
        litho_class_index = int(litho_class_value)
        vol_colorname = self.color_names[litho_class_index]
        class_name = self.class_names[litho_class_index]
        s = extract_single_lithology_class_3d(lithology_3d_array, litho_class_value)
        vol_mesh = (self.xxx, self.yyy, self.zzz, s)
        self.overlay_bore_classes(self.dem_mesh, vol_mesh, self.bore_data, vol_colorname, z_label='AHD x 50', points_scale_factor=self.POINTS_SCALE_FACTOR, title='Lithology class: ' + class_name)

def get_colorscale_lut(vis_widget):
    # not sure this is valid for all mayavi things...
    return vis_widget.module_manager.scalar_lut_manager.lut


def prep_proba_for_contour(lithology_3d_proba):
    xp = np.copy(lithology_3d_proba)
    other_value = 0.0
    # We burn the edges of the volume, as I suspect this is necessary to have a more intuitive viz (otherwuse non closed volumes)
    xp[0,:,:] = other_value
    xp[-1,:,:] = other_value
    xp[:,0,:] = other_value
    xp[:,-1,:] = other_value
    xp[:,:,0] = other_value
    xp[:,:,-1] = other_value
    return xp


@mlab.show
def render_proba_contour(prob_volume, color_name, proba_level = 0.5, title=None):
    s = prep_proba_for_contour(prob_volume)
    mlab.figure(size=(800, 800))
    mlab.contour3d(s, contours=[proba_level], color=to_rgb(color_name))
    mlab.outline()
    mlab.xlabel(EASTING_COL)
    mlab.ylabel(NORTHING_COL)
    mlab.zlabel('mAHD')
    if not title is None:
        mlab.title(title)