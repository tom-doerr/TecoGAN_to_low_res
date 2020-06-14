#!/usr/bin/env python3
import imageio
import glob
import numpy as np
import shutil

SCENE_LR_DIRS_PATH = './LR/'
DIR_NAME_TARGET = 'calendar'
DIR_NAME_SOURCE = 'calendar_original'
DIR_NAME_BEST = 'calendar_best'
METRICS_FILE_PATH = 'results/metric_log/metrics.csv'

def get_rand_tensor(shape):
    rand_tensor_not_rounded = np.random.rand(*shape)
    rand_tensor_rounded = np.around(rand_tensor_not_rounded)
    return rand_tensor_rounded

def get_score():
    pass


dir_source = SCENE_LR_DIRS_PATH + DIR_NAME_SOURCE 
dir_best = SCENE_LR_DIRS_PATH + DIR_NAME_BEST
shutil.copytree(dir_source, dir_best)

for im_path_source in glob.glob(SCENE_LR_DIRS_PATH + DIR_NAME_SOURCE + '/*.png'):
     image = imageio.imread(im_path)
     rand_tensor = get_rand_tensor(image.shape)
     image_new = image + rand_tensor
     im_path_target = im_path_source.replace(DIR_NAME_SOURCE, DIR_NAME_TARGET)
     imageio.imwrite(im_path_target, image_new)
