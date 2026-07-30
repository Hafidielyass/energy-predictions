#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

strace -e trace=openat -f python -c "
import tensorflow as tf
tf.config.list_physical_devices('GPU')
" > /tmp/strace_out.txt 2>&1

grep -E "libcu|libnv|ENOENT" /tmp/strace_out.txt | grep -v "\.pyc\|\.py\b" | head -60
