#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== Current numpy version ==="
python -c "import numpy; print(numpy.__version__)"

echo ""
echo "=== Upgrading pyarrow, scikit-learn for numpy 2.x compatibility ==="
pip install "pyarrow>=15.0" "scikit-learn>=1.5" 2>&1 | tail -10

echo ""
echo "=== Verifying TF imports ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
import pandas as pd
print('pandas:', pd.__version__)
import scipy
print('scipy:', scipy.__version__)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All\|WARNING:tensorflow"

echo ""
echo "=== Verifying JAX GPU still works ==="
python -c "
import jax
print('JAX devices:', jax.devices())
" 2>&1 | grep -v "WARNING: All\|oneDNN"
