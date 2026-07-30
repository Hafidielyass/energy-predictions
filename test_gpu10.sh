#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# Write stderr to file to capture pre-absl messages
python -c "
import tensorflow as tf
tf.config.list_physical_devices('GPU')
" 2>/tmp/tf_stderr.txt
cat /tmp/tf_stderr.txt
echo "---FULL STDERR ABOVE---"
