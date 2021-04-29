
```sh
my_env_name=ela

# python 3.9 seems there is a breaking change and mayavi use of notebook rendering.
# Nope, creates a 3.9 still...
# conda env create -n $my_env_name -f /home/per202/src/ela/pyela/trials/environment_nover.yml python=3.8
conda create -c conda-forge -n $my_env_name python=3.8
conda activate ela
conda install -c conda-forge mamba

mamba install -c conda-forge  rasterio cartopy geopandas pandas nltk scikit-learn matplotlib vtk wordcloud pyqt mayavi pip ipykernel ipywidgets ipyevents

mamba install -c conda-forge jupyterlab # (because we need to install the labextensions???) http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks


jupyter lab build

python -m ipykernel install --user --name ${my_env_name} --display-name "Lithology"

# maybe need striplog fork of mine??
# pip install my_striplog      - git+https://github.com/jmp75/striplog@master#egg

pip install ela

mamba install -c conda-forge xlrd
mamba install -c conda-forge openpyxl

# data_proj=ccrs.epsg(28350)
# ModuleNotFoundError: No module named 'pyepsg'
mamba install -c conda-forge pyepsg

sudo mkdir -p ${HOME}/mnt/piwi
sudo mount -t cifs //fs1-per.nexus.csiro.au/{lw-piwi}/work/ ${HOME}/mnt/piwi -o user=per202,domain=nexus

```

