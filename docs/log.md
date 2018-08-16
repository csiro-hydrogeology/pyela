# Log

## 2018-08-16

TIL, and a rant.
Trying to document the installation of pyela using conda, creating a supposedly clean conda env. But tells me some packages are alreadh installed (vtk). To my dismay the site-packages directory of another conda environment is in the sys.path of the new 'test' environment:

```python
>>> sys.path
['', '/home/per202/anaconda3/envs/ELA_TEST/lib/python36.zip', '/home/per202/anaconda3/envs/ELA_TEST/lib/python3.6', '/home/per202/anaconda3/envs/ELA_TEST/lib/python3.6/lib-dynload', '/home/per202/.local/lib/python3.6/site-packages', '/home/per202/src/github_jm/spotpy/build/lib', '/home/per202/src/github_jm/striplog', '/home/per202/anaconda3/envs/ELA/lib/python3.6/site-packages/python_dateutil-2.7.3-py3.6.egg', '/home/per202/anaconda3/envs/ELA/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg', '/home/per202/anaconda3/envs/ELA/lib/python3.6/site-packages/affine-2.2.1-py3.6.egg', '/home/per202/anaconda3/envs/ELA/lib/python3.6/site-packages/munch-2.3.2-py3.6.egg', '/home/per202/anaconda3/envs/ELA/lib/python3.6/site-packages', '/home/per202/anaconda3/envs/ELA_TEST/lib/python3.6/site-packages']
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

```sh
# Note: not sure if conda-forge needed: conda config --add channels conda-forge
conda config --show-sources
conda config --remove channels conda-forge
```

```sh
my_env_name=ELA
```

```sh
conda env remove --name ${my_env_name}
```

Trying to install ela requirements with `python install -s requirements.txt`:

```
Collecting Cartopy>=0.16.0 (from -r requirements.txt (line 9))
    ModuleNotFoundError: No module named 'numpy'
    ImportError: NumPy 1.6+ is required to install cartopy.
```

Seems I need to install each requirement (numpy, cython) with `pip install` anyway beforehand. Bit useless.

