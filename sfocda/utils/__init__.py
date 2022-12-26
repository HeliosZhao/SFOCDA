import pathlib
from .func import *

project_root = pathlib.Path(__file__).resolve().parents[2]

__all__ = ['project_root']
