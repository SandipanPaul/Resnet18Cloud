#!/bin/bash
sudo apt install python3.10-venv
#Create virtual environment
VENV_NAME=".venv"
python3 -m venv "$VENV_NAME"

echo "Activating virtual environment"
. .venv/bin/activate

#download the project dependencies
pip3 install -r requirements.txt
pip3 install git+https://github.com/reconfigurable-ml-pipeline/load_tester

#Download images from git
if [ -d "images" ]; then
    rm -rf images
fi
mkdir images
cd images
git clone https://github.com/EliSchwartz/imagenet-sample-images

