
Refreshing the notes to install `ela` as of 2021-04. The context of the dependencies has changed, apparently slightly better in fact. 

## windows

```bat
call C:\Users\per202\AppData\Local\Continuum\anaconda3\Scripts\activate.bat
set my_env_name=ela
conda create -c conda-forge -n %my_env_name%  python=3.8
conda activate %my_env_name% 
conda install -c conda-forge mamba
mamba install -c conda-forge  rasterio cartopy geopandas pandas nltk scikit-learn matplotlib vtk wordcloud pyqt mayavi pip ipykernel ipywidgets ipyevents
mamba install -c conda-forge jupyterlab 

:: following may be optional,, but does not hurt
jupyter lab build
python -m ipykernel install --user --name %my_env_name% --display-name "Lithology"

:: data_proj=ccrs.epsg(28350)
:: ModuleNotFoundError: No module named 'pyepsg', and other new dependencies
:: so:
mamba install -c conda-forge xlrd openpyxl pyepsg

pip install ela
```

```bat
call C:\Users\per202\AppData\Local\Continuum\anaconda3\Scripts\activate.bat
call conda activate ELA
:: setting ELA_SRC probably optional
:: set ELA_SRC=c:\src\github_jm\pyela
:: mapping "\\fs1-cbr.nexus.csiro.au\{lw-digwat-1}\work" to the Z: drive
set ELA_DATA=Z:\lithology
cd C:\src\github_jm\pyela-doc\tutorials
call jupyter-lab .
:: Opening getting_started seems to run including 3d vis
```

## Linux

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

# xlrd openpyxl required to load some of the datasets in PIWI
# data_proj=ccrs.epsg(28350)
# ModuleNotFoundError: No module named 'pyepsg'
mamba install -c conda-forge xlrd openpyxl pyepsg

sudo mkdir -p ${HOME}/mnt/piwi
sudo mount -t cifs //fs1-per.nexus.csiro.au/{lw-piwi}/work/ ${HOME}/mnt/piwi -o user=per202,domain=nexus

```

## 3D display on dual graphics optimus laptops

Trying to use on Linux a nvidia card, on a new laptop. Notebook kernel started with "primusrun": mayavi displays a black screen and in the notebook there are warnings such as:

```text
2021-07-25 20:57:21.095 (1057.616s) [        F8CB1740]     vtkOpenGLState.cxx:1380  WARN| Hardware does not support the number of textures defined. 
```

Installed the nvidia proprietary driers (from nvidia managed additional repo urls, not the 'non-free' stuff in official repositories). Using primus. See notes in /home/per202/src/rr-ml/lstm/readme.md. 

`primusrun glxgears` works as expected so far as I can tell, so the failure with mayavi rentering is a mystery. Lots of arcane things can go wrong.


Dell has in its KB: https://www.dell.com/support/kbdoc/en-au/000132622/a-guide-to-nvidia-optimus-on-dell-pcs-with-an-ubuntu-operating-system

https://help.ubuntu.com/community/BinaryDriverHowto/Nvidia

An informed post with an FAQ section is https://archived.forum.manjaro.org/t/black-screen-after-trying-to-install-proprietary-drivers-on-manjaro/90104/7

Do I need to (and can) do https://www.dell.com/community/Precision-Mobile-Workstations/Disable-Intel-UHD-in-BIOS-and-set-Nvidia-Quadro-as-primary/td-p/7338385  ?

In passing, I find the laptop a bit laggy. Hint: https://www.dell.com/community/Precision-Mobile-Workstations/Precision-7550-Degraded-Performance-After-BIOS-Update/td-p/7926086


Trying to disalbe graphics switching mode at the bios level. I enable the Direct Output mode thinggy (*but this is unclear what this is for)

Can boot, but systems warns that there is no hardware accelleration and cpu usage may be high.

dmesg output has :

```text
[    3.765890] nvidia-modeset: Loading NVIDIA Kernel Mode Setting Driver for UNIX platforms  470.42.01  Tue Jun 15 21:22:38 UTC 2021
[    3.933184] nvidia-modeset: WARNING: GPU:0: BOE Technology Group Co., Ltd (DP-4): G-SYNC Compatible: EDID min refresh rate invalid, disabling G-SYNC Compatible.
```

also, perhaps more benign:

```text
[   99.927436] nvidia_uvm: module uses symbols from proprietary module nvidia, inheriting taint.
[   99.930821] nvidia-uvm: Loaded the UVM driver, major device number 238.
```

```text
[    2.786260] nvidia: loading out-of-tree module taints kernel.
[    2.786267] nvidia: module license 'NVIDIA' taints kernel.
[    2.786268] Disabling lock debugging due to kernel taint
[    2.822386] nvidia: module verification failed: signature and/or required key missing - tainting kernel
```

nvidia-settings gives an error.

https://forums.developer.nvidia.com/t/nvidia-settings-error-unable-to-load-info-from-any-available-system/172872/8

https://forums.lenovo.com/t5/Ubuntu/Ubuntu-on-Lenovo-ThinkPad-p15g-with-Nvidia-2070-MQ/m-p/5052835

https://wiki.archlinux.org/title/Variable_refresh_rate

https://bbs.archlinux.org/viewtopic.php?id=258201  
