# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import re
import os
from os import getcwd
from os.path import exists, isdir, isfile, join
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import progressbar
import logging
import gc
from utils.utils import *

class ImageString(object):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, img_filename):
        self.img_filename = img_filename
        self.patient = self._parse_patient()
        self.study = self._parse_study()
        self.image_num = self._parse_image()
        self.study_type = self._parse_study_type()
        self.image = self._parse_image()
        self.normal = self._parse_normal()
        self.encode = self._parse_encode()

    def flat_file_name(self):
        return "{}_{}_patient{}_study{}_image{}.png".format(self.normal, self.study_type, self.patient, self.study,
                                                            self.image, self.normal)

    def _parse_patient(self):
        return int(self._patient_re.search(self.img_filename).group(1))

    def _parse_study(self):
        return int(self._study_re.search(self.img_filename).group(1))

    def _parse_image(self):
        return int(self._image_re.search(self.img_filename).group(1))

    def _parse_study_type(self):
        return self._study_type_re.search(self.img_filename).group(1)

    def _parse_normal(self):
        return "normal" if ("negative" in self.img_filename) else "abnormal"

    def _parse_encode(self):
        return 0 if ("negative" in self.img_filename) else 1

# copy only Train or Validation set: train, val, or all
copy_data_type = 'train'
target_size = 224
data_path='/home/walte/data'

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# processed
# data
# ├── train
# │   ├── abnormal
# │   └── normal
# └── val
#     ├── abnormal
#     └── normal

## sample weights
#Elbow:   0.6237172178  0.3762827822
#Finger:  0.661498708   0.338501292
#Hand:    0.7418235877  0.2581764123
#Humerus: 0.5422297297  0.4577702703
#Forearm: 0.6727480046  0.3272519954
#Shoulder:0.4835164835  0.5164835165
#Wrist:   0.6167630058  0.3832369942
abnormal_weights =  {'ELBOW':0.62, 'FINGER':0.66, 'FOREARM':0.67, 'HAND':0.74, 'HUMERUS':0.54, 'SHOULDER':0.48, 'WRIST':0.62}
normal_weights =  {'ELBOW':0.38, 'FINGER':0.34, 'FOREARM':0.33, 'HAND':0.26, 'HUMERUS':0.46, 'SHOULDER':0.52, 'WRIST':0.38}

# make new folder to hold the pickle files
proc_data_dir = join(data_path, 'MURA-data')

#obsolete: re-organize images files to new structure
#proc_train_dir = join(proc_data_dir, 'train')
#proc_val_dir = join(proc_data_dir, 'val')
#if not os.path.exists(proc_data_dir):
#    os.mkdir(proc_data_dir)
#if not os.path.exists(proc_train_dir):
#    os.mkdir(proc_train_dir)
#if not os.path.exists(proc_val_dir):
#    os.mkdir(proc_val_dir)


# Data loading code
orig_data_dir = join(data_path, 'MURA-v1.1')
train_dir = join(orig_data_dir, 'train')
train_csv = join(orig_data_dir, 'train_image_paths.csv')
val_dir = join(orig_data_dir, 'valid')
val_csv = join(orig_data_dir, 'valid_image_paths.csv')
test_dir = join(orig_data_dir, 'test')
assert isdir(orig_data_dir) and isdir(train_dir) and isdir(val_dir) and isdir(test_dir)
assert exists(train_csv) and isfile(train_csv) and exists(val_csv) and isfile(val_csv)

#local variables
img_list = []
label_list = []
sample_weights = []

if copy_data_type in ('train', 'all'):
    log.info(f'loading images from {train_csv}') 
    sample_type = 'train'
    df = pd.read_csv(train_csv, names=['img', 'label'], header=None)
    samples = [tuple(x) for x in df.values]

    i = 0
    with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
        for img, label in samples:
            #assert ("negative" in img) is (label is 0)
            if ("negative" in img) and (label is 0):
                print (img)
                print (label)
                assert ("negative" in img) is (label is 0)
            img1 = join(data_path, img)
            enc = ImageString(img1)
            #read image
            new_img = Image.open(enc.img_filename).convert('RGB')
            # resize and convert it to array
            img_list.append(np.asarray(new_img.resize(
                    (target_size, target_size), Image.ANTIALIAS)))
            label_list.append(enc.encode)
            if enc.normal == 'normal':
                s_weight = normal_weights.get(enc.study_type)
            else:
                s_weight = abnormal_weights.get(enc.study_type)
            sample_weights.append(s_weight)
            i += 1
            bar.update(i)
            
    log.info('stacking images')
    x = np.stack(img_list, axis=0)
    # forcing garbage collection to save memory
    del img_list; gc.collect()
    y = np.stack(label_list, axis=0)
    del label_list; gc.collect()
    w = np.stack(sample_weights, axis=0)
    del sample_weights; gc.collect()
    
    log.info('pickling images')
    pkl_fn = join(proc_data_dir, f'x_{sample_type}.pkl')
    log.info('build picker file name: {pkl_fn}')
    write_pickle_file(x, pkl_fn) 
    del x; gc.collect()
    pkl_fn = join(proc_data_dir, f'y_{sample_type}.pkl')
    log.info('build picker file name: {pkl_fn}')
    write_pickle_file(y, pkl_fn) 
    del y; gc.collect()
    pkl_fn = join(proc_data_dir, f'w_{sample_type}.pkl')
    log.info('build picker file name: {pkl_fn}')
    write_pickle_file(w, pkl_fn) 
    del w; gc.collect()

# obsolete code: copy data to a new folder with flat_file_name
#if copy_data_type in ('val', 'all'):
#    df = pd.read_csv(val_csv, names=['img', 'label'], header=None)
#    samples = [tuple(x) for x in df.values]
#    for img, label in samples:
#        #assert ("negative" in img) is (label is 0)
#        img1 = join(data_path, img)
#        enc = ImageString(img1)
#        cat_dir = join(proc_val_dir, enc.normal)
#        if not os.path.exists(cat_dir):
#            os.mkdir(cat_dir)
#        shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))

img_list = []
label_list = []
sample_weights = []

if copy_data_type in ('val', 'all'):
    log.info(f'loading images from {val_csv}') 
    sample_type = 'valid'
    df = pd.read_csv(val_csv, names=['img', 'label'], header=None)
    samples = [tuple(x) for x in df.values]
    # progress bar
    i = 0
    with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
        for img, label in samples:
            #assert ("negative" in img) is (label is 0)
            if ("negative" in img) and (label is 0):
                print (img)
                print (label)
                assert ("negative" in img) is (label is 0)
            img1 = join(data_path, img)
            enc = ImageString(img1)
            #read image
            new_img = Image.open(enc.img_filename).convert('RGB')
            # resize and convert it to array
            img_list.append(np.asarray(new_img.resize(
                    (target_size, target_size), Image.ANTIALIAS)))
            label_list.append(enc.encode)
            if enc.normal == 'normal':
                s_weight = normal_weights.get(enc.study_type)
            else:
                s_weight = abnormal_weights.get(enc.study_type)
            sample_weights.append(s_weight)
            i += 1
            bar.update(i)
            
    log.info('stacking images')
    x = np.stack(img_list, axis=0)
    # forcing garbage collection to save memory
    del img_list; gc.collect()
    y = np.stack(label_list, axis=0)
    del label_list; gc.collect()
    w = np.stack(sample_weights, axis=0)
    del sample_weights; gc.collect()
    
    log.info('pickling images')
    pkl_fn = join(proc_data_dir, f'x_{sample_type}.pkl')
    log.info('build picker file name: {pkl_fn}')
    write_pickle_file(x, pkl_fn) 
    del x; gc.collect()
    pkl_fn = join(proc_data_dir, f'y_{sample_type}.pkl')
    log.info('build picker file name: {pkl_fn}')
    write_pickle_file(y, pkl_fn) 
    del y; gc.collect()
    pkl_fn = join(proc_data_dir, f'w_{sample_type}.pkl')
    log.info('build picker file name: {pkl_fn}')
    write_pickle_file(w, pkl_fn) 
    del w; gc.collect()
