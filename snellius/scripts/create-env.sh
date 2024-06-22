#!/bin/bash

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $HOME/pyreft

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/stanfordnlp/pyreft.git
