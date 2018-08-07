from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from PIL import Image, ImageOps
import pickle
import gc
import logging
from utils.utils import *
# add RegEx
import re

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def read_pickle2array(path='/home/walte/data/MURA-data', sample=None, y_output=3):
    '''
    Get data from pickle files

    Parameters
    ----------
    path: str
        path to pickled MURA data
    sample: str
        if None, then read both train and valid sample and
        return a tuple of 4 elements
        otherwise, return a tuple of two elements corresponding
        to the x and y datasets for the sample.

    Return
    ------
    `obj`:tuple of `obj`:numpy.ndarray
    '''
    assert sample in (None, 'train', 'valid'), (
        "sample must be one of (None, 'train', 'valid')")
    #output can be either 2 or 3, if missing default to 3
    assert y_output in (2, 3), (
        "sample must be one of (2, 3)")
        
    mura_path = Path(path)

    if not sample or sample == 'train':
        train_X = read_pickle_file(mura_path.joinpath(f'x_train.pkl'))
        train_Y = read_pickle_file(mura_path.joinpath(f'y_train.pkl'))
        if y_output == 3:
            train_W = read_pickle_file(mura_path.joinpath(f'w_train.pkl'))

        if sample == 'train':
            if y_output == 2:
                return train_X, train_Y
            else:
                return train_X, train_Y, train_W
    if not sample or sample == 'valid':
        test_X = read_pickle_file(mura_path.joinpath(f'x_valid.pkl'))
        test_Y = read_pickle_file(mura_path.joinpath(f'y_valid.pkl'))
        if y_output == 3:
            test_W = read_pickle_file(mura_path.joinpath(f'w_valid.pkl'))

        if sample == 'valid':
            if y_output == 2:
                return test_X, test_Y
            else:
                return test_X, test_Y, test_W
    if y_output == 2:
        return train_X, train_Y, test_X, test_Y
    else:
        return train_X, train_Y, train_W, test_X, test_Y, test_W


