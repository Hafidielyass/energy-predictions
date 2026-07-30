#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

python -c "
# Reproduce exactly what configure_tf does, step by step
import os, logging, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Import JAX first
import jax
print('JAX devices:', jax.devices())

# Now import TF
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF GPUs:', gpus)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All\|WARNING:jax"
