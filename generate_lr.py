#!/usr/bin/env python3
import imageio
import glob
import numpy as np
import shutil
import os

SCENE_LR_DIRS_PATH = './LR/'
DIR_NAME_TARGET = 'calendar'
DIR_NAME_SOURCE = 'calendar_original'
DIR_NAME_BEST = 'calendar_best'
METRICS_FILE_PATH = 'results/metric_log/metrics.csv'
EVALUATE_UPSAMPLING_COMMAND = '''
docker run \
--gpus all -it \
--mount src=$(pwd),target=/TecoGAN,type=bind \
-w /TecoGAN tecogan_image \
bash -c "
python3 runGan.py 2
"
'''

def get_rand_tensor(shape):
    rand_tensor_not_rounded = np.random.rand(*shape)
    rand_tensor_rounded = np.around(rand_tensor_not_rounded)
    return rand_tensor_rounded

def get_score():
    metrics_content = ''
    with open(METRICS_FILE_PATH, 'r') as f:
        metrics_content = f.read()

    print("metrics_content:", metrics_content)
    last_line_metrics = metrics_content.split('\n')[-2]
    print("last_line_metrics:", last_line_metrics)
    FrameAvg_tOF = last_line_metrics.split(',')[-1]
    print("FrameAvg_tOF:", FrameAvg_tOF)
    return float(FrameAvg_tOF)

def execute_evaluation_upsampling():
    os.system(EVALUATE_UPSAMPLING_COMMAND + ' > /dev/null')

def evaluate_lr_images():
    execute_evaluation_upsampling()
    return get_score()

def add_noise_to_image(image):
     rand_tensor = get_rand_tensor(image.shape)
     image_new = image + rand_tensor
     return image_new

def add_noise_to_images(dir_name_source, dir_name_target):
    for im_path_source in glob.glob(dir_name_source + '/*.png'):
         image = imageio.imread(im_path_source)
         image_new = add_noise_to_image(image)
         image_path_target = im_path_source.replace(dir_name_source, dir_name_target)
         imageio.imwrite(image_path_target, image_new)


if __name__ == '__main__':
    dir_source = SCENE_LR_DIRS_PATH + DIR_NAME_SOURCE 
    dir_best = SCENE_LR_DIRS_PATH + DIR_NAME_BEST
    dir_target = SCENE_LR_DIRS_PATH + DIR_NAME_TARGET
    shutil.rmtree(dir_best)
    shutil.copytree(dir_source, dir_best)
    shutil.rmtree(dir_target)
    shutil.copytree(dir_best, dir_target)
    initial_score = evaluate_lr_images()
    best_score = initial_score

    while True:
        add_noise_to_images(DIR_NAME_BEST, DIR_NAME_TARGET)
        score = evaluate_lr_images()
        if score > best_score:
            best_score = score
            shutil.copytree(dir_target, dir_best)
        print(f'score: {score}   improvement: {best_score - initial_score}')



    for im_path_source in glob.glob(SCENE_LR_DIRS_PATH + DIR_NAME_SOURCE + '/*.png'):
         image = imageio.imread(im_path_source)
         im_path_target = im_path_source.replace(DIR_NAME_SOURCE, DIR_NAME_TARGET)
         imageio.imwrite(im_path_target, image_new)
