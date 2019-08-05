# Releasing - notes

## 2019-01-07

Dry run for submission to pypi. Bank on learnings from [releasing refcount](https://github.com/jmp75/didactique/blob/master/doc/know_how.md#python-packaging-for-pypi)

```bash
cd ~/src/github_jm/pyela
source ~/anaconda3/bin/activate
```

```bash
my_env_name=ELA
conda activate ${my_env_name}
conda install wheel twine six pytest
```

update Readme and the "_version.py" file


```bash
rm dist/*
pandoc -f markdown -t rst README.md  > README.rst
python3 setup.py sdist bdist_wheel
```

Importantly to not end up with incorrect display of the readme:

```bash
twine check dist/*
```

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Then and only then:

```bash
twine upload dist/*
```

## Troubleshooting

```bash
pandoc -f markdown -t rst README.md  > README.rst
```

Can view with the `retext` program (did not find VScode RST extensions working, or giving out blank output if not, perhaps)

```bash
python setup.py check --restructuredtext
```

If you have a markdown readme, you may want to convert with `pandoc` and in setup.py use:

```python
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    long_description_content_type='text/x-rst'
```

`twine check` may pass with the markdown, but after pandoc conversion picked up the following:

```text
warning: Check: The project's long_description has invalid markup which will not be rendered on PyPI. The following syntax errors were detected:
line 146: Warning: Cannot analyze code. No Pygments lexer found for "txt".
```

## 2018-09-16

Idea: consider using "binder" for interactive tut. See [spaCy doc](https://spacy.io/)

```cmd
call C:\Users\XXXYYY\AppData\Local\Continuum\anaconda3\Scripts\activate.bat
set my_env_name=ELA3
conda create --name %my_env_name% python=3.6
REM later : conda install --name %my_env_name% xarray netCDF4 rasterio cartopy jupyterlab ipywidgets jupyter geopandas pandas scikit-learn scikit-image matplotlib python=3.6
conda install --name %my_env_name% jupyterlab ipywidgets jupyter pandas scikit-learn matplotlib python=3.6
conda activate %my_env_name%
python -m ipykernel install --user --name %my_env_name% --display-name "Python3 (ELA)"
conda install --name %my_env_name% spaCy
python -m spacy download en
```

```cmd
py3env.bat
activate ELA3
python -m spacy download en
```

```text
python -m spacy download en
Installing collected packages: en-core-web-sm
  Running setup.py install for en-core-web-sm ... done
Successfully installed en-core-web-sm-2.0.0
You are using pip version 10.0.1, however version 18.0 is available.
You should consider upgrading via the 'python -m pip install --upgrade pip' command.

    Error: Couldn't link model to 'en'
    Creating a symlink in spacy/data failed. Make sure you have the required
    permissions and try re-running the command as admin, or use a
    virtualenv. You can still import the model as a module and call its
    load() method, or create the symlink manually.
```

```cmd
cd C:\src\github_jm\pyela\trials
py3env.bat
activate ELA3
jupyter-lab
```

`nlp = spacy.load('en_core_web_sm')` or `nlp = spacy.load('en')` still fail.

### Troubleshoot

TIL:

If you launch a kernel from jupyter but you do not get the right environment, you may have registered the kernel from the incorrect env.. Check e.g. C:\Users\XXXYYY\AppData\Roaming\jupyter\kernels\ela3\kernel.json:

`"C:\\Users\\XXXYYY\\AppData\\Local\\Continuum\\anaconda3\\python.exe"` where you would need `"C:\\Users\\XXXYYY\\AppData\\Local\\Continuum\\anaconda3\\envs\\ELA3\\python.exe"`

## 2018-08-16

TIL, and a rant.
Trying to document the installation of pyela using conda, creating a supposedly clean conda env. But tells me some packages are alreadh installed (vtk). To my dismay the site-packages directory of another conda environment is in the sys.path of the new 'test' environment:

```python
>>> sys.path
['', '/home/XXXYYY/anaconda3/envs/ELA_TEST/lib/python36.zip', '/home/XXXYYY/anaconda3/envs/ELA_TEST/lib/python3.6', '/home/XXXYYY/anaconda3/envs/ELA_TEST/lib/python3.6/lib-dynload', '/home/XXXYYY/.local/lib/python3.6/site-packages', '/home/XXXYYY/src/github_jm/spotpy/build/lib', '/home/XXXYYY/src/github_jm/striplog', '/home/XXXYYY/anaconda3/envs/ELA/lib/python3.6/site-packages/python_dateutil-2.7.3-py3.6.egg', '/home/XXXYYY/anaconda3/envs/ELA/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg', '/home/XXXYYY/anaconda3/envs/ELA/lib/python3.6/site-packages/affine-2.2.1-py3.6.egg', '/home/XXXYYY/anaconda3/envs/ELA/lib/python3.6/site-packages/munch-2.3.2-py3.6.egg', '/home/XXXYYY/anaconda3/envs/ELA/lib/python3.6/site-packages', '/home/XXXYYY/anaconda3/envs/ELA_TEST/lib/python3.6/site-packages']
>>>
```

WTF??? 

```
~/.local/lib/python3.6/site-packages$ more easy-install.pth 
blah/src/github_jm/spotpy/build/lib
blah/src/github_jm/striplog
blah/anaconda3/envs/ELA/lib/python3.6/site-packages/python_dateutil-2.7.3-py3.6.egg
blah/anaconda3/envs/ELA/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg
blah/anaconda3/envs/ELA/lib/python3.6/site-packages/affine-2.2.1-py3.6.egg
blah/anaconda3/envs/ELA/lib/python3.6/site-packages/munch-2.3.2-py3.6.egg
```

```
:~/.local/lib/python3.6/site-packages$ more setuptools.pth 
blah/anaconda3/envs/ELA/lib/python3.6/site-packages
```

So, lesson, I think: beware of doing `python setup.py install --user` and/or with pip, from a conda environment. I think that is what messes things up.

```bash
# Note: not sure if conda-forge needed: conda config --add channels conda-forge
conda config --show-sources
conda config --remove channels conda-forge
```

```bash
my_env_name=ELA
```

```bash
conda env remove --name ${my_env_name}
```

Trying to install ela requirements with `python install -s requirements.txt`:

```
Collecting Cartopy>=0.16.0 (from -r requirements.txt (line 9))
    ModuleNotFoundError: No module named 'numpy'
    ImportError: NumPy 1.6+ is required to install cartopy.
```

Seems I need to install each requirement (numpy, cython) with `pip install` anyway beforehand. Bit useless.

