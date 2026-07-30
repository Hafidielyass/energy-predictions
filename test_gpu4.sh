#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
unset TF_CPP_MIN_LOG_LEVEL
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1
