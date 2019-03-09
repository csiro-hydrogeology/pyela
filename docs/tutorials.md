

An occasion to also recreate from scratch a conda env for pyela. Update documentation.

Downloaded from the BoM NGIS the Murrumbidgee data in shapefile form.
DEM manually defined area, from http://elevation.fsdf.org.au/
downloading an area east of bungendore, srtm 1 second DEM. Saving as geotiff, coord sys GDA 1994 epsg 4283 

```bash
cd ~/data/
mkdir ela
cd ela
curl -o one_second_dem_bungendore.zip http://download.elvis.ga.gov.au.s3.amazonaws.com/CLIP_17629.zip
unzip one_second_dem_bungendore.zip

conda install --name ELA jupyterlab ipywidgets jupyter

# nodejs already installed from synaptic, debian repo
jupyter-labextension install @jupyter-widgets/jupyterlab-manager
jupyter-labextension install ipyvolume
jupyter-labextension install jupyter-threejs

python3 -m ipykernel install --user --name ELA --display-name "Python3 (ELA)"

```

Idea: using [gmaps in notebooks](https://jupyter-gmaps.readthedocs.io/en/latest/install.html#installing-jupyter-gmaps-for-jupyterlab)

Note: for the EDA if reading xls files using pandas, need pkg `xlrd`

```bash
source ~/anaconda3/bin/activate
conda activate ELA
cd /home/$USER/src/github_jm/pyela/docs/tutorials
jupyter-lab .
```