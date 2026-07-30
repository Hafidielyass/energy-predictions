#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

python -c "
# Try loading JAX CUDA plugin BEFORE TF to see if TF picks it up
import jax
jax.config.update('jax_platform_name', 'cuda')
print('JAX devices:', jax.devices())

import tensorflow as tf
print('TF GPUs (after JAX):', tf.config.list_physical_devices('GPU'))

# Try explicit PJRT registration
try:
    tf.experimental.dtensor.initialize_accelerator_system()
    print('dtensor init OK')
except Exception as e:
    print('dtensor init failed:', e)

print('TF GPUs (after dtensor):', tf.config.list_physical_devices('GPU'))
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All"

echo ""
echo "=== Check TF PJRT plugin search ==="
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
# List all registered devices
devices = tf.config.list_physical_devices()
print('All physical devices:', devices)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All"
