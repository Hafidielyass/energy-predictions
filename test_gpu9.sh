#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# Try with XLA_FLAGS verbose and TF_XLA_FLAGS
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_VMODULE=gpu_device=5,cuda_driver=5 \
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import tensorflow as tf
print('TF:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
# Also try XLA
try:
    from jax.lib import xla_bridge
    print('JAX backend:', xla_bridge.get_backend().platform)
    import jax
    print('JAX devices:', jax.devices())
except Exception as e:
    print('JAX check failed:', e)
" 2>&1 | grep -v "^DEBUG\|^INFO" | grep -i "gpu\|cuda\|device\|error\|warn\|fail\|skip\|physical\|jax" | head -40
