"""A package for Exploratory Lithology Analysis.

See:
https://github.com/jmp75/pyela
"""

from setuptools import setup
import re
# To use a consistent encoding
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

verstr = 'unknown'
VERSIONFILE = "ela/_version.py"
with open(VERSIONFILE, "r")as f:
    verstrline = f.read().strip()
    pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
    mo = pattern.search(verstrline)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# try:
#     import pypandoc
#     long_description = pypandoc.convert('README.md', 'rst')
#     long_description = long_description.replace("\r","") # Do not forget this line
#     long_description_content_type='text/markdown'
# except:
#     print("Pandoc not found. Long_description conversion failure.")
#     # Get the long description from the README file
#     with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#         long_description = f.read()
#         long_description_content_type='text/markdown'

# with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()
#     long_description_content_type='text/markdown'

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    long_description_content_type='text/x-rst'



# with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as f:
#     REQUIREMENTS = f.read().splitlines()

REQUIREMENTS = ['pandas',
                'geopandas',
                'nltk',
                'numpy',
                'cartopy',
                'mayavi',
                'worldcloud',
                'rasterio',
                'scikit-learn',
                'scipy',
                'striplog',  # NOTE should be git+https://github.com/jmp75/striplog@master#egg=striplog for 2.7, I think.
                'matplotlib'
                ]

TEST_REQUIREMENTS = ['pytest',
                     'coveralls',
                     'pytest-cov',
                     'pytest-mpl'
                     ]

CLASSIFIERS = ['Development Status :: 3 - Alpha',
                'Intended Audience :: Science/Research',
                'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
                'Programming Language :: Python',
                # 'Programming Language :: Python :: 2.7',  # see striplog caveat above
                'Programming Language :: Python :: 3.6'
                ]
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='ela',
    version=verstr,
    description='A Python package for exploratory lithology analysis', 
    long_description=long_description,
    url='https://github.com/jmp75/pyela',
    author='Jean-Michel Perraud, Juan Castilla',
    author_email='per202@csiro.au',
    classifiers=CLASSIFIERS,
    keywords='lithology geophysics groundwater data-mining',
    packages=['ela', 'ela.experiment'],
    install_requires=REQUIREMENTS,
    extras_require={
        ':python_version >= "3.6"': [
            'PyQt5',
        ]
    # extras_require={  # Optional
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    # package_data={  # Optional
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jmp75/pyela/issues',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/jmp75/pyela',
    },
)
