ela
===

Exploratory Lithology Analysis
------------------------------

`license <https://github.com/jmp75/pyela/blob/devel/LICENSE.txt>`__
|status| master: `build <https://travis-ci.org/jmp75/pyela>`__
`coverage <https://coveralls.io/github/jmp75/pyela?branch=master>`__

`Getting started tutorial - Bungendore 3D
lithology <https://github.com/jmp75/pyela-doc/blob/master/tutorials/getting_started.ipynb>`__

Analysing driller’s logs is a tedious and repetitive task in many
groundwater modelling projects. Automating the process of extracting
useful information from driller’s logs allows spending less time on
manual data wrangling, more time on its interpretation, and enhances the
reproducibility of the analysis.

This package combines features to:

-  perform natural language processing on lithology descriptions in the
   logs, to detect primary and secondary lithologies
-  apply supervised machine learning to interpolate lithologies across a
   3D grid
-  visualise interactively the 3D data

License
-------

MIT (see
`License.txt <https://github.com/jmp75/pyela/blob/master/LICENSE.txt>`__)

Documentation
-------------

Tutorials are hosted `in a separate github
repo <https://github.com/jmp75/pyela-doc>`__

Get a `quick tour of the visualisation part of
‘ela’ <https://github.com/jmp75/pyela/blob/master/docs/visual_tour.md>`__.

Installation
------------

As of January 2019 `ela is on pypi <https://pypi.org/project/ela/>`__.
Before installing you may want to skim the “Installation details”
section thereafter to think of context (e.g. conda env) but otherwhse
``pip install ela`` should install the required dependencies.

.. code:: bash

   pip search ela
   pip install ela

Alternatively, from source:

.. code:: bash

   pip install -r requirements.txt
   python setup.py install

Installation details
--------------------

‘ela’ relies on several external packages, and some can be fiddly to
install depending on the version of Python and these external packages.
This section thus has fairly prescriptive instructions, given in the
hope of limiting the risk of issues.

The first part of this section is focused on Linux.

Debian packages for spatial projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cartopy`` and possibly other python packages require ``proj4`` version
4.9+ to be installed (libproj-dev). If your debian/ubuntu repo does not
suffice (older versions) you may try:

.. code:: bash

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

Installation of python packages dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to install
`Anaconda <http://docs.continuum.io/anaconda/install>`__ to install
dependencies. Note that I recommend to **not** let anaconda change your
startup file and change the ``PATH`` environment. To activate Anaconda
you first need: ``source ~/anaconda3/bin/activate``. Then choose a conda
environment name.

Optionally you may want to do ``conda update -n base conda`` and
``conda update -n base anaconda-navigator``

.. code:: bash

   my_env_name=ELA

.. code:: bash

   conda create --name ${my_env_name} python=3.6
   conda activate  ${my_env_name}
   conda install --name ${my_env_name} rasterio cartopy geopandas pandas nltk scikit-learn scikit-image matplotlib vtk

As of writing (2018-08) conda does not have pyqt5, and a suitable
version of mayavi for python3. We use ``pip``

.. code:: bash

   pip install --upgrade pip

For Python 3.x one needs to install pyqt5 for mayavi, as per `these
instructions <https://docs.enthought.com/mayavi/mayavi/installation.html>`__.
As of Jan 2019 be aware that there is a `known issue in mayavi visual
rendering with pyqt5 as a
backend <https://github.com/enthought/mayavi/issues/656>`__ and ‘ela’ is
affected by this. Nevertheless this is not a commplete blocker for most
‘ela’ features so installation instructions are kept here.

.. code:: bash

   pip search pyqt5 | sort -g
   pip search mayavi

.. code:: bash

   pip install pyqt5
   pip install mayavi

Windows
~~~~~~~

Placeholder section. As of Sept 2018 it may be possible to install upon
Python 3.6+ with Anaconda 3, and then including mayavi from pip.

Related Geoscience packages
---------------------------

‘ela’ aims to complement other Python packages for geoscience, in
particular for handling bore data . It depends on the package
`‘striplog’ <https://github.com/agile-geoscience/striplog>`__ and is
likely to depend on `‘lasio’ <https://github.com/kinverarity1/lasio>`__
in the future.

Known issues
------------

3D interactive visualisation: As of 2018-08, using mayavi 4.6 on python
3.6 may be `visually
buggy <https://github.com/enthought/mayavi/issues/656>`__, though users
reported success on 2019-03. Python 2.7 with mayavi 4.5 via Anaconda2 is
known to work.

Troubleshooting
---------------

If in a conda environment trying to use ``pip`` you get:

.. code:: text

   ModuleNotFoundError: No module named 'pip._internal'

consider:

.. code:: bash

   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python3 get-pip.py --force-reinstall

.. |status| image:: https://img.shields.io/badge/status-alpha-blue.svg

