#!/bin/bash

##___________________ INSTALLATION GUIDE ------------


conda create -n my_emulator python=3.11
conda activate my_emulator
python -m pip install -r requirements.txt
conda install -c conda-forge jupyterlab notebook ipykernel -y
python -m pip install --no-build-isolation --no-deps git+https://github.com/justinalsing/affine.git

#______________________________________________________
