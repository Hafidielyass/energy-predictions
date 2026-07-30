#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
python -c "
import os, ctypes
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH','(not set)'))
# Test loading by name (relies on LD_LIBRARY_PATH)
for lib in ['libcudart.so.12','libcublas.so.12','libcufft.so.11','libcudnn.so.9']:
    try:
        ctypes.CDLL(lib); print(f'OK: {lib}')
    except Exception as e: print(f'FAIL: {lib}: {e}')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF:', tf.__version__)
print('GPUs:', gpus)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild"
