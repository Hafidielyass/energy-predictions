#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF:', tf.__version__)
print('GPUs:', gpus)
if gpus:
    print('SUCCESS - RTX 4070 detected!')
else:
    print('FAIL - no GPU found')
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild TensorFlow"
