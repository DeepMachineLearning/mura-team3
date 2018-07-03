import logging
from pathlib import Path
import pickle
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def normalize_pixels(x):
    x.astype("float32")
    x /= 255
    return x

def pre_process (X, Y, target_size ):
   
    images = []
    height,width = target_size
    
    log.info('prprocess images')
    images = []
    for image in X:
        image = image.resize(size = (height,width))
        image = np.asarray(image)
        image.astype('float32')
        image = image/255
        images.append(image)

    
    x_images = np.stack(images, axis=0)
    y_labels = np.stack(Y, axis=0)
    
    return x_images, y_labels 
        

    
def write_pickle_file(obj_to_pickle, pickle_file_path):
    with Path(pickle_file_path).open('wb') as pick_file:
        log.info(f'saving to {Path(pickle_file_path).as_posix()}')
        pickle.dump(obj_to_pickle, pick_file, protocol=4)

def read_pickle_file( pickle_file_path ):
    with Path(pickle_file_path).open('rb') as pickle_file:
        log.info(f'loading {Path(pickle_file_path).as_posix()}')
        return pickle.load(pickle_file)