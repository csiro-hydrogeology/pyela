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

## Related Geoscience packages

'ela' aims to complement other Python packages for geoscience, in particular for handling bore data . It depends on the package ['striplog'](https://github.com/agile-geoscience/striplog) and is likely to depend on ['lasio'](https://github.com/kinverarity1/lasio) in the future.


## Documentation

A [quick tour of a workflow with pyela](https://localhost:80)

Placeholder for introductory material such as tutorials

<!-- See here for the [complete ela package documentation](https://ela.readthedocs.io/en/latest/). -->

<!-- Draft Notes
-----------

``conda install coveralls`` then ``conda install pytest-cov pytest-mpl``: this downgrades numpy a tad. Why? Trying ``conda update numpy`` but even odder outcome -->

## Quick start

Install the usual way:

```bash
$ pip install ela
```

Very quick example session:

```python
>>> import ela
>>> las = ela.read("sample_big.las")
```

Data is accessible both directly as numpy arrays

```python
>>> las.keys()
['DEPT', 'DT', 'RHOB', 'NPHI', 'SFLU', 'SFLA', 'ILM', 'ILD']
>>> las['SFLU']
array([ 123.45,  123.45,  123.45, ...,  123.45,  123.45,  123.45])
>>> las['DEPT']
array([ 1670.   ,  1669.875,  1669.75 , ...,  1669.75 ,  1670.   ,
        1669.875])
```

and as ``CurveItem`` objects with associated metadata:

```python
>>> las.curves
[CurveItem(mnemonic=DEPT, unit=M, value=, descr=1  DEPTH, original_mnemonic=DEPT, data.shape=(29897,)), 
CurveItem(mnemonic=DT, unit=US/M, value=, descr=2  SONIC TRANSIT TIME, original_mnemonic=DT, data.shape=(29897,)), 
CurveItem(mnemonic=RHOB, unit=K/M3, value=, descr=3  BULK DENSITY, original_mnemonic=RHOB, data.shape=(29897,)), 
CurveItem(mnemonic=NPHI, unit=V/V, value=, descr=4   NEUTRON POROSITY, original_mnemonic=NPHI, data.shape=(29897,)), 
CurveItem(mnemonic=SFLU, unit=OHMM, value=, descr=5  RXO RESISTIVITY, original_mnemonic=SFLU, data.shape=(29897,)), 
CurveItem(mnemonic=SFLA, unit=OHMM, value=, descr=6  SHALLOW RESISTIVITY, original_mnemonic=SFLA, data.shape=(29897,)), 
CurveItem(mnemonic=ILM, unit=OHMM, value=, descr=7  MEDIUM RESISTIVITY, original_mnemonic=ILM, data.shape=(29897,)), 
CurveItem(mnemonic=ILD, unit=OHMM, value=, descr=8  DEEP RESISTIVITY, original_mnemonic=ILD, data.shape=(29897,))]
```

Header information is parsed into simple HeaderItem objects, and stored in a dictionary for each section of the header:

```python
>>> las.version
[HeaderItem(mnemonic=VERS, unit=, value=1.2, descr=CWLS LOG ASCII STANDARD -VERSION 1.2, original_mnemonic=VERS), 
HeaderItem(mnemonic=WRAP, unit=, value=NO, descr=ONE LINE PER DEPTH STEP, original_mnemonic=WRAP)]
>>> las.well
[HeaderItem(mnemonic=STRT, unit=M, value=1670.0, descr=, original_mnemonic=STRT), 
HeaderItem(mnemonic=STOP, unit=M, value=1660.0, descr=, original_mnemonic=STOP), 
HeaderItem(mnemonic=STEP, unit=M, value=-0.125, descr=, original_mnemonic=STEP), 
HeaderItem(mnemonic=NULL, unit=, value=-999.25, descr=, original_mnemonic=NULL), 
HeaderItem(mnemonic=COMP, unit=, value=ANY OIL COMPANY LTD., descr=COMPANY, original_mnemonic=COMP), 
HeaderItem(mnemonic=WELL, unit=, value=ANY ET AL OIL WELL #12, descr=WELL, original_mnemonic=WELL), 
HeaderItem(mnemonic=FLD, unit=, value=EDAM, descr=FIELD, original_mnemonic=FLD), 
HeaderItem(mnemonic=LOC, unit=, value=A9-16-49, descr=LOCATION, original_mnemonic=LOC), 
HeaderItem(mnemonic=PROV, unit=, value=SASKATCHEWAN, descr=PROVINCE, original_mnemonic=PROV), 
HeaderItem(mnemonic=SRVC, unit=, value=ANY LOGGING COMPANY LTD., descr=SERVICE COMPANY, original_mnemonic=SRVC), 
HeaderItem(mnemonic=DATE, unit=, value=25-DEC-1988, descr=LOG DATE, original_mnemonic=DATE), 
HeaderItem(mnemonic=UWI, unit=, value=100091604920, descr=UNIQUE WELL ID, original_mnemonic=UWI)]
>>> las.params
[HeaderItem(mnemonic=BHT, unit=DEGC, value=35.5, descr=BOTTOM HOLE TEMPERATURE, original_mnemonic=BHT), 
HeaderItem(mnemonic=BS, unit=MM, value=200.0, descr=BIT SIZE, original_mnemonic=BS), 
HeaderItem(mnemonic=FD, unit=K/M3, value=1000.0, descr=FLUID DENSITY, original_mnemonic=FD), 
HeaderItem(mnemonic=MATR, unit=, value=0.0, descr=NEUTRON MATRIX(0=LIME,1=SAND,2=DOLO), original_mnemonic=MATR), 
HeaderItem(mnemonic=MDEN, unit=, value=2710.0, descr=LOGGING MATRIX DENSITY, original_mnemonic=MDEN), 
HeaderItem(mnemonic=RMF, unit=OHMM, value=0.216, descr=MUD FILTRATE RESISTIVITY, original_mnemonic=RMF), 
HeaderItem(mnemonic=DFD, unit=K/M3, value=1525.0, descr=DRILL FLUID DENSITY, original_mnemonic=DFD)]
```

The data is stored as a 2D numpy array:

```python
>>> las.data
array([[ 1670.   ,   123.45 ,  2550.   , ...,   123.45 ,   110.2  ,   105.6  ],
       [ 1669.875,   123.45 ,  2550.   , ...,   123.45 ,   110.2  ,   105.6  ],
       [ 1669.75 ,   123.45 ,  2550.   , ...,   123.45 ,   110.2  ,   105.6  ],
       ...,
       [ 1669.75 ,   123.45 ,  2550.   , ...,   123.45 ,   110.2  ,   105.6  ],
       [ 1670.   ,   123.45 ,  2550.   , ...,   123.45 ,   110.2  ,   105.6  ],
       [ 1669.875,   123.45 ,  2550.   , ...,   123.45 ,   110.2  ,   105.6  ]])
```

You can also retrieve and load data as a ``pandas`` DataFrame, build LAS files from scratch, 
write them back to disc, and export to Excel, amongst other things.

See the [documentation](https://ela.readthedocs.io/en/latest/) for more details.

## License

MIT
