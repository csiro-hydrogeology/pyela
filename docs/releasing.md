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
```

Normally commits on the testing branch would trigger tests on travis, but no harm in local run. Remember you need to have done `pip install codecov coverage pytest-cov pytest-mpl cython`

```bash
coverage run -m pytest
```

```bash
pandoc -f markdown -t rst README.md  > README.rst
python3 setup.py sdist bdist_wheel
```

Importantly to reduce the risl of ending up with incorrect display of the readme.rst on pypi:

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

## doc generation

Found out that I needed to use the napoleon extention to read google style docstrings

```bash
conda install sphinx

sphinx-quickstart


cd ~/src/ela/pyela
sphinx-apidoc -f -o ./docs/source ela

cd docs 
make html
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

