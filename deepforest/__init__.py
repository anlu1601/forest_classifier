# -*- coding: utf-8 -*-
"""Top-level package for DeepForest."""
__author__ = """Ben Weinstein"""
__email__ = 'ben.weinstein@weecology.org'
__version__ = '0.3.2'

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
print("ROOT:", _ROOT)


def get_data(path):
    return os.path.join(_ROOT, 'deepforest/data', path)


def get_eval(path):
    return os.path.join(_ROOT, "\\model\\eval", path)


def get_train(path):
    return os.path.join(_ROOT, "\\model\\train", path)


def get_train_dir():
    return os.path.join(_ROOT, "\\model\\train")


def get_output(path):
    return os.path.join(_ROOT, "output", path)


def get_output_dir():
    return os.path.join(_ROOT, "output")


def get_model(path):
    return os.path.join(_ROOT, "model", path)


def get_model_dir():
    return os.path.join(_ROOT, "model")
