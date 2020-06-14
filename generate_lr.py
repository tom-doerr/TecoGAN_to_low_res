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
DIR_NAME_SECOND_BEST = 'calendar_best2'
METRICS_FILE_PATH = 'results/metric_log/metrics.csv'
ORIGINAL_IMAGES_PATH = 'HR/calendar/'
OUTPUT_PATH_DIR = 'results/calendar/'
EVALUATE_UPSAMPLING_COMMAND = '''
docker run \
--gpus all -it \
--mount src=$(pwd),target=/TecoGAN,type=bind \
-w /TecoGAN tecogan_image \
bash -c "
python3 runGan.py 2 \
 > /dev/null 2>&1
"
'''

INFERENCE_COMMAND = '''
docker run \
--gpus all -it \
--mount src=$(pwd),target=/TecoGAN,type=bind \
-w /TecoGAN tecogan_image \
bash -c "
python3 runGan.py 1 \
 > /dev/null 2>&1
"
'''                            

def get_rand_tensor(shape):
    rand_tensor_not_rounded = np.random.rand(*shape)
    rand_tensor_rounded = np.around(rand_tensor_not_rounded)
    #rand_tensor_uint8 = np.array(rand_tensor_rounded, dtype=np.uint8)
    return rand_tensor_rounded

def get_score():
    metrics_content = ''
    with open(METRICS_FILE_PATH, 'r') as f:
        metrics_content = f.read()

    last_line_metrics = metrics_content.split('\n')[-2]
    FrameAvg_tOF = last_line_metrics.split(',')[-1]
    return float(FrameAvg_tOF)

def execute_evaluation_upsampling():
    os.system(EVALUATE_UPSAMPLING_COMMAND + '')

def evaluate_lr_images():
    execute_evaluation_upsampling()
    return get_score()

def add_noise_to_image(image):
    rand_tensor = get_rand_tensor(image.shape)
    image_new_not_clipped = np.array(image, dtype=np.int16) + rand_tensor
    image_new = np.array(np.clip(image_new_not_clipped, a_min=0, a_max=255), dtype=np.uint8)
    return image_new

def execute_inference():
    os.system(INFERENCE_COMMAND)

def calculate_l1_distance_images(dir_name_source):
    l1_distance = 0
    for im_path_source in glob.glob(dir_name_source + '*.png'):
         image_original = imageio.imread(im_path_source)
         image_reconstructed_path = im_path_source.replace(dir_name_source, OUTPUT_PATH_DIR+'output_')
         image_reconstructed = imageio.imread(image_reconstructed_path)
         l1_distance += np.sum(image_reconstructed - image_original)

    return l1_distance


def add_noise_to_images(dir_name_source, dir_name_target):
    for im_path_source in glob.glob(SCENE_LR_DIRS_PATH + dir_name_source + '/*.png'):
         image = imageio.imread(im_path_source)
         image_new = add_noise_to_image(image)
         image_path_target = im_path_source.replace(dir_name_source, dir_name_target)
         imageio.imwrite(image_path_target, image_new)


if __name__ == '__main__':
    dir_source = SCENE_LR_DIRS_PATH + DIR_NAME_SOURCE 
    dir_best = SCENE_LR_DIRS_PATH + DIR_NAME_BEST
    dir_second_best = SCENE_LR_DIRS_PATH + DIR_NAME_SECOND_BEST
    dir_target = SCENE_LR_DIRS_PATH + DIR_NAME_TARGET
    shutil.rmtree(dir_best)
    shutil.copytree(dir_source, dir_best)
    shutil.rmtree(dir_target)
    shutil.copytree(dir_best, dir_target)
    #initial_score = evaluate_lr_images()
    execute_inference()
    initial_score = calculate_l1_distance_images(ORIGINAL_IMAGES_PATH)
    best_score = initial_score

    while True:
        shutil.rmtree(dir_target)
        os.mkdir(dir_target)
        add_noise_to_images(DIR_NAME_BEST, DIR_NAME_TARGET)
        #score = evaluate_lr_images()
        execute_inference()
        score = calculate_l1_distance_images(ORIGINAL_IMAGES_PATH)
        if score < best_score:
            best_score = score
            shutil.rmtree(dir_second_best)
            shutil.copytree(dir_best, dir_second_best)
            shutil.rmtree(dir_best)
            shutil.copytree(dir_target, dir_best)
        print(f'score: {score}   initial_score: {initial_score}   improvement: {initial_score - best_score}')



