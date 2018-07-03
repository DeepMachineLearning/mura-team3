import logging
import progressbar 
import pandas as pd 
from PIL import Image 
import os
from pathlib import Path 
import numpy as np
from utils.util import *



log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def write_raw_mura2pickle(data_path = 'data/MURA-v1.1',bone_name = 'ALL',sample = None):
    
    assert sample in ['train', 'valid', 'test'] 
    assert bone_name in ['ALL', 'XR_ELBOW', 'XR_FINGER', 'FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    
    file = f'{sample}_image_paths.csv'
    full_path = os.path.join(data_path, file)
    
    root_path = os.path.basename(os.path.dirname(data_path))

    image_list = []
    label_list = [] 
    
    i = 0

    paths = pd.read_csv(full_path, header = None)  
    paths.columns = ['path']
    
    if bone_name == 'ALL':
        log.info(f'reading all images from {sample}')
        image_path_list = paths['path']
    else :
        log.info(f"reading images from {sample}, bone name: {bone_name}")
        image_path_list = [path for path in paths['path'] if f'{bone_name}' in path]
    

    with progressbar.ProgressBar(max_value = len(image_path_list)) as bar:
        for image_file in image_path_list:
            label = 1 if 'positive' in image_file else 0 
            img = Image.open(os.path.join(root_path, image_file)).convert('L')
            #img = np.array(img)
            
            image_list.append(img)
            label_list.append(label)

            i += 1
            bar.update(i)
    x_pickle_path = os.path.join(data_path, f'{sample}_x_{bone_name}.pkl')
    write_pickle_file(obj_to_pickle=image_list, pickle_file_path= x_pickle_path)
    log.info('writing x_pickcle finishes')

    y_pickle_path = os.path.join(data_path, f'{sample}_y_{bone_name}.pkl')
    write_pickle_file(obj_to_pickle=label_list, pickle_file_path= y_pickle_path)
    log.info('writing y_pickle finishes')
    
    return image_list, label_list
 






def read_raw_mura_pickle(path = 'data/MURA-v1.1', sample = None, bone_name = 'ALL'):
    
    assert sample in ['train', 'valid', 'test'] 
    assert bone_name in ['ALL', 'XR_ELBOW', 'XR_FINGER', 'FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    
    x_str = f'{path}/{sample}_x_{bone_name}.pkl'
    y_str = f'{path}/{sample}_y_{bone_name}.pkl'
    ret_x = os.path.exists(x_str) 
    ret_y = os.path.exists(y_str)

    if not ret_x or not ret_y:
        log.info(f'{x_str} or {y_str} does not exist')
        os._exit(1)
    else: 
        return read_pickle_file(x_str), read_pickle_file(y_str)
        

    '''
    Reads mura datasets, pads with 0's to a square image,
    resizes each edge to be target_size, and then saves
    the datasets as pickle files
    
    Parameters
    ----------
    mura_path: str
        string path of MURA-v1.1
    sample: str
        switch for running for train or valid sample
    target_size: int
        target edge length for resizing
        
    Returns
    -------
    None
    
    '''
def load_mura_data(data_path = 'data/MURA-v1.1', sample = 'valid', bone_name = 'ALL', target_size = (256,256)):
    assert sample in ['train', 'valid', 'test'] 
    assert bone_name in ['ALL', 'XR_ELBOW', 'XR_FINGER', 'FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    

    # read the picle file if it exists, otherwise write
    '''
    str = f'{data_path}/{sample}_x_{bone_name}.pkl'
    ret = os.path.exists(str) 
    if not ret :
        log.info(f'{str} does not exist')
        write_pickle_file()   # I need to add 
    else: 
        return read_pickle_file(str)
    '''
    x_str = f'{data_path}/{sample}_x_{bone_name}.pkl'
    y_str = f'{data_path}/{sample}_y_{bone_name}.pkl'
    ret_x = os.path.exists(x_str) 
    ret_y = os.path.exists(y_str)

    if  ret_x and ret_y:
        log.info(f'{x_str} or {y_str} does not exist')
        X_image, Y_labels   = read_raw_mura_pickle(data_path, sample, bone_name)  
    else: 
        X_image, Y_labels  = write_raw_mura2pickle(data_path, bone_name, sample)
    
    X_image, Y_labels = pre_process(X_image, Y_labels, target_size)    
    return X_image, Y_labels
         

    
if __name__ == '__main__':
    X, Y = load_mura_data(data_path='data/MURA-v1.1', bone_name='XR_FINGER',sample='valid',target_size=(256, 256))
    