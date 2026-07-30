#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== Current TF install ==="
pip show tensorflow | grep -E "Name|Version|Location"

echo ""
echo "=== Reinstalling tensorflow[and-cuda] ==="
pip install "tensorflow[and-cuda]==2.17.0" 2>&1 | tail -10

echo ""
echo "=== Verify ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All"
