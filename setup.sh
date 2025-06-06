#!/bin/bash

#Create virtual environment
VENV_NAME=".venv"
python3 -m venv "$VENV_NAME"
. .venv/bin/activate

#download the project dependencies
pip3 install -r requirements.txt

#Download images from git
mkdir images
cd images
git clone https://github.com/EliSchwartz/imagenet-sample-images
pip3 install git+https://github.com/reconfigurable-ml-pipeline/load_tester
