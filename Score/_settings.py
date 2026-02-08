import getpass
import os
import sys

__USERNAME = getpass.getuser()

# Paths relative to this package (Score/), so they work when run from repo root or Score
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.join(_THIS_DIR, 'data')
MODEL_PATH = os.path.join(_BASE_DIR, 'weights')
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

