ela
===

Exploratory Lithology Analysis

|license| |status| master: |build| |coverage|

Analysing driller’s logs is a tedious and repetitive task in many
groundwater modelling projects. Automating the process of extracting
useful information from driller's logs allows spending less time on
manual data wrangling, more time on its interpretation, and enhances the
reproducibility of the analysis.

This package combines features to:

-  perform natural language processing lithology descriptions in the
   logs, to detect primary and secondary lithologies
-  apply supervised machine learning to interpolate lithologies across a
   3D grid
-  visualise interactively the 3D data

License
-------

MIT (see `License.txt <./LICENSE.txt>`__)

Documentation
-------------

Get a `quick tour of the visualisation part of
'ela' <./docs/visual_tour.md>`__

Installation
------------

Note that 'ela' relies on several external packages, and some can be
fiddly to install depending on the version of Python and packages. Below
are fairly prescriptive instructions, given in the hope of limiting the
risk of issues.

Debian packages for spatial projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cartopy`` and possibly other python packages require ``proj4`` version
4.9+ to be installed (libproj-dev). If your debian/ubuntu repo does not
suffice (older versions) you may try:

.. code:: sh

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

Installation of python dependencies with conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to install
`Anaconda <http://docs.continuum.io/anaconda/install>`__ to install
dependencies. Note that I recommend to **not** let anaconda change your
startup file and change the ``PATH`` environment. To activate Anaconda
you first need: ``source ~/anaconda3/bin/activate``. Then choose a conda
environment name.

Optionally you may want to do ``conda update -n base conda`` and
``conda update -n base anaconda-navigator``

.. code:: sh

    my_env_name=ELA

.. code:: sh

    conda create --name ${my_env_name} python=3.6
    conda activate  ${my_env_name}
    conda install --name ${my_env_name} rasterio cartopy geopandas pandas nltk scikit-learn scikit-image matplotlib vtk

As of writing (2018-08) conda does not have pyqt5, and a suitable
version of mayavi for python3. We use ``pip``

.. code:: sh

    pip install --upgrade pip

.. code:: sh

    pip search pyqt5
    pip search mayavi

.. code:: sh

    pip install pyqt5
    pip install mayavi

Windows
~~~~~~~

Placeholder section. As of Sept 2018 it may be possible to install upon
Python 3.6+ with Anaconda 3, and then including mayavi from pip.

Installation of pyela
~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install -r requirements.txt
    python setup.py install

For Python 2.7.x pyqt5 is not available:

.. code:: sh

    # Note: not sure if conda-forge needed: conda config --add channels conda-forge
    conda create --name  ${my_env_name} python=2.7 mayavi rasterio cartopy geopandas pandas nltk scikit-learn scikit-image matplotlib vtk

Related Geoscience packages
---------------------------

'ela' aims to complement other Python packages for geoscience, in
particular for handling bore data . It depends on the package
`'striplog' <https://github.com/agile-geoscience/striplog>`__ and is
likely to depend on `'lasio' <https://github.com/kinverarity1/lasio>`__
in the future.

Known issues
------------

As of 2018-08, using mayavi 4.6 on python 3.6 is `buggy, a VTK issue it
seems <https://github.com/enthought/mayavi/issues/656>`__. Python 2.7
with mayavi 4.5 via Anaconda2 is known to work.

Troubleshooting
---------------

If in a conda environment trying to use ``pip`` you get:

.. code:: text

    ModuleNotFoundError: No module named 'pip._internal'

consider:

.. code:: sh

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --force-reinstall

.. |license| image:: http://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/jmp75/pyela/blob/devel/LICENSE.txt
.. |status| image:: https://img.shields.io/badge/status-alpha-blue.svg
.. |build| image:: https://img.shields.io/travis/jmp75/pyela.svg?branch=master
   :target: https://travis-ci.org/jmp75/pyela
.. |coverage| image:: https://coveralls.io/repos/github/jmp75/pyela/badge.svg?branch=master
   :target: https://coveralls.io/github/jmp75/pyela?branch=master
