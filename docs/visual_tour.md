# A visual quick tour of 'ela'

This section is a quick tour of ela using some of its visual outputs.

## 3D visualisations with mayavi

<!-- http://richardstudynotes.blogspot.com/2014/04/link-images-stored-in-google-drive-to.html -->
3D interactive visualisation is an important way to get insight into complicated volumetric data such as bore data and the volumes of properties derived from the exploration and interpolation performed with 'ela'. This section gives a sample from a workflow using ela to determine hydraulic conductivities over an underground volume of interest.

The most likely conductivity class over a 3D gridded volume can be visualised primarily using planar transect and volumetric rendering.

![Hydraulic conductivity class - planar transect](https://drive.google.com/uc?id=10d4nTykp9BwBl1jt7RQ1m9PrlsrH_YLm "Hydraulic conductivity class - planar transect")

![Hydraulic conductivity class - volume rendering of the most likely class](https://drive.google.com/uc?id=1gEQuBSqtSB4O4q_xNruw6RgVViEoa7Wn "Hydraulic conductivity class - volume rendering of the most likely class")

For a given conductivity class, it is possible with 'ela' to get a field of probability, resulting from the supervised machine learning with 'scikit'. Again volumetric planar transect and volumetric rendering.

![Hydraulic conductivity class - planar transect of probabilities](https://drive.google.com/uc?id=1C0gmxvIto2g3RqFByBxXYbh5R5ineHI_ "Hydraulic conductivity class - planar transect of probabilities")

![Hydraulic conductivity class - Isosurface of a probability field for a class](https://drive.google.com/uc?id=1SUnAK_OVX4EEPyCHkSxkn1i5b4-t6cFi "Hydraulic conductivity class - Isosurface of a probability field for a class")

'ela' has custom facilities built on top of mayavi to overlay geometrically heterogeneous but spatially related data: elevation raster, volume of classes, and information from the bore data:

![Hydraulic conductivity class - Overlay bore data, terrain and interpolated class](https://drive.google.com/uc?id=1cvdUaQ6bc6AmePNaAG-OIieboMazmvvf "Hydraulic conductivity class - Overlay bore data, terrain and interpolated class")
<!-- NOTE: see https://about.gitlab.com/handbook/product/technical-writing/markdown-guide/#display-other-videos -->

<!-- blank line -->
<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/1pWsqPYdb9s_u7-y4wl6RqH2BJcxYH76C/preview" frameborder="0" allowfullscreen="true" width="1200" height="757"> </iframe>
</figure>
<!-- blank line -->

## 2D visualisations with matplotlib

Part of ela is about the extraction of lithology terms from bore data. It has facility to assess the frequency of some terms using frequency plots.

![Word frequency in lithology descriptions - terms derived from 'clay'](https://drive.google.com/uc?id=1dookOYjLNUiy9RafRGY2Hlw_W-kvqKCG "Hydraulic conductivity class - Overlay bore data, terrain and interpolated class")

Placeholder for a visual with cartopy