#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "---"
# Check what TF C-level sees
python -c "
import os, subprocess
result = subprocess.run(['python', '-c',
    'import os; print(os.environ.get(\"LD_LIBRARY_PATH\",\"not set\"))'],
    env=os.environ, capture_output=True, text=True)
print('Child LD_LIBRARY_PATH:', result.stdout.strip())
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All"
