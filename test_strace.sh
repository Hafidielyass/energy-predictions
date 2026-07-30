#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

strace -e trace=openat,open -f python -c "
import tensorflow as tf
tf.config.list_physical_devices('GPU')
" 2>&1 | grep -E "libcu|libcudnn|libnv|ENOENT.*lib" | grep -v "\.pyc\|__pycache__\|\.py\"" | head -50
