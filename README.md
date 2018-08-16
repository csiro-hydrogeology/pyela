# ela

![Status](https://img.shields.io/badge/status-alpha-orange.svg)
[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jmp75/pyela/blob/devel/LICENSE.txt)
[![Build Status](https://img.shields.io/travis/jmp75/pyela.svg?branch=devel)](https://travis-ci.org/jmp75/pyela)

<!-- [![Docker Build](https://img.shields.io/docker/build/kinverarity/ela.svg)](https://hub.docker.com/r/kinverarity/ela/)
[![Build status](https://ci.appveyor.com/api/projects/status/csr7bg8urkbtbq4n?svg=true)](https://ci.appveyor.com/project/kinverarity1/ela)
[![Python versions](https://img.shields.io/pypi/pyversions/ela.svg)](https://www.python.org/downloads/) -->
<!-- [![Version](http://img.shields.io/pypi/v/ela.svg)](https://pypi.python.org/pypi/ela/) -->

Python package for exploratory lithology analysis

Analysing driller’s logs is a tedious and repetitive task in many groundwater modelling projects. Automating the process of extracting useful information from driller's logs allows spending less time on manual data wrangling, more time on its interpretation, and enhances the reproducibility of the analysis.

This packages combines features to:

* perform natural language processing lithology descriptions in the logs, to detect primary and secondary lithologies
* apply supervised machine learning to interpolate lithologies across a 3D grid
* visualise interactively the 3D data

## Documentation

Placeholder for introductory material such as tutorials, including a [placeholder for a quick tour of a workflow with pyela](https://localhost:80)

<!-- See here for the [complete ela package documentation](https://ela.readthedocs.io/en/latest/). -->

<!-- Draft Notes
-----------

``conda install coveralls`` then ``conda install pytest-cov pytest-mpl``: this downgrades numpy a tad. Why? Trying ``conda update numpy`` but even odder outcome -->

## Installation

### Debian packages for spatial projections

`cartopy` and possibly other python packages require `proj4` version 4.9+ to be installed (libproj-dev). If your debian/ubuntu repo does not suffice (older versions) you may try:

```sh
sudo apt-get install -y libc6  
wget http://en.archive.ubuntu.com/ubuntu/pool/universe/p/proj/proj-data_4.9.3-2_all.deb
sudo dpkg -i proj-data_4.9.3-2_all.deb
wget http://en.archive.ubuntu.com/ubuntu/pool/universe/p/proj/libproj12_4.9.3-2_amd64.deb
sudo dpkg -i libproj12_4.9.3-2_amd64.deb
wget http://en.archive.ubuntu.com/ubuntu/pool/universe/p/proj/proj-bin_4.9.3-2_amd64.deb
sudo dpkg -i proj-bin_4.9.3-2_amd64.deb
wget http://en.archive.ubuntu.com/ubuntu/pool/universe/p/proj/libproj9_4.9.2-2_amd64.deb 
sudo dpkg -i libproj9_4.9.2-2_amd64.deb
wget http://en.archive.ubuntu.com/ubuntu/pool/universe/p/proj/libproj-dev_4.9.3-2_amd64.deb
sudo dpkg -i libproj-dev_4.9.3-2_amd64.deb
```

### Installation of python dependencies with conda

You may want to install [Anaconda](http://docs.continuum.io/anaconda/install) to install dependencies. Note that I recommend to **not** let anaconda change your startup file and change the `PATH` environment. To activate Anaconda you first need: `source ~/anaconda3/bin/activate`. Then choose a conda environment name.

Optionally you may want to do `conda update -n base conda` and `conda update -n base anaconda-navigator`

```sh
my_env_name=ELA
```

```sh
conda create --name ${my_env_name} python=3.6
conda activate  ${my_env_name}
conda install --name ${my_env_name} rasterio cartopy geopandas pandas nltk scikit-learn scikit-image matplotlib vtk
```

As of writing (2018-08) conda does not have pyqt5, and a suitable version of mayavi for python3. We use `pip`

```sh
pip install --upgrade pip
```

```sh
pip search pyqt5
pip search mayavi
```

```sh
pip install pyqt5
pip install mayavi
```

### Installation of pyela

```sh
pip install -r requirements.txt
python setup.py install
```

For Python 2.7.x pyqt5 is not available:

```sh
# Note: not sure if conda-forge needed: conda config --add channels conda-forge
conda create --name  ${my_env_name} python=2.7 mayavi rasterio cartopy geopandas pandas nltk scikit-learn scikit-image matplotlib vtk
```

## License

MIT

## Troubleshooting

If in a conda environment trying to use `pip` you get:

```txt
ModuleNotFoundError: No module named 'pip._internal'
```

consider:

```sh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --force-reinstall
```