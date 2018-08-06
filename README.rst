ela
========

.. image:: https://img.shields.io/travis/jmp75/pyela.svg
    :target: https://travis-ci.org/jmp75/pyela
    :alt: Travis build status

Python package for exploratory lithology analysis

Analysing driller’s logs is a tedious and repetitive task in many groundwater modelling projects. Automating the process of extracting useful information from driller's logs allows spending less time on manual data wrangling, more time on its interpretation, and enhances the reproducibility of the analysis.

This packages combines features to: 

    * perform natural language processing lithology descriptions in the logs, to detect primary and secondary lithologies
    * apply supervised machine learning to interpolate lithologies across a 3D grid
    * visualise interactively the 3D data 

Docs
----

* Placeholder


Dependencies
------------

Most are best installed with Anaconda, see **Install**, below.

* `NumPy <http://www.numpy.org/>`_
* `matplotlib <http://matplotlib.org/>`_


Install
-------

* ``pip install ela``

I recommend setting up a virtual environment:

* Install `Anaconda <http://docs.continuum.io/anaconda/install>`_ if you don't have it already
* Then do this to create an environment called ``myenv`` (or whatever you like), answering Yes to the confirmation question::

    conda create -n myenv python=3.5 numpy matplotlib
    source activate myenv

* Then you can do::

    pip install ela


Development: setting up for testing
-----------

There are other requirements for testing, as listed in ``setup.py``. They should install with::

    python setup.py test

But I had better luck doing ``conda install pytest`` first.

The tests can be run with::

    python run_tests.py


Development: running the bleeding edge
-----------

To run the latest version of the code, you should be on the `develop` branch:

    git clone https://github.com/jmp75/ela.git
    cd ela
    git checkout develop
    
You probably want to continue in your virtual environment (see above).

Then I use these commands, which you can join with `;` if you like, to keep the software up to date:

    /usr/bin/yes | pip uninstall ela     # Of course you don't need this one if you didn't install it yet.
    python setup.py sdist
    pip install dist/ela-0.6.1.tar.gz    # Or whatever was the last version to build.

Draft Notes
-----------

``conda install coveralls`` then ``conda install pytest-cov pytest-mpl``: this downgrades numpy a tad. Why? Trying ``conda update numpy`` but even odder outcome