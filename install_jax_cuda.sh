#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== Installing JAX with CUDA 12 support ==="
pip install -U "jax[cuda12]" 2>&1 | tail -15

echo ""
echo "=== Verifying JAX GPU ==="
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
" 2>&1 | grep -v "^I0000\|WARNING: All\|oneDNN\|instructions"

echo ""
echo "=== Verifying TF GPU ==="
python -c "
import tensorflow as tf
print('TF version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All"
