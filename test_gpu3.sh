#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
# Unset log suppression to see full GPU init messages
unset TF_CPP_MIN_LOG_LEVEL
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1 | grep -i "gpu\|cuda\|driver\|dlopen\|cannot\|error\|warning\|device\|skip" | head -40
