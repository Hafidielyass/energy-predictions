#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== TF package location ==="
python -c "import tensorflow as tf; print(tf.__file__)"

echo "=== GPU plugin files ==="
find /home/msi/miniconda3/envs/tf_gpu -name "*gpu*" -o -name "*cuda*" 2>/dev/null | grep -v "__pycache__\|\.pyc" | grep "\.so\|plugin\|pywrap" | head -20

echo "=== What TF's _pywrap_tensorflow_internal links to ==="
python -c "
import tensorflow as tf, os
tf_dir = os.path.dirname(tf.__file__)
import glob
libs = glob.glob(tf_dir + '/**/*.so*', recursive=True)
for l in libs[:5]:
    print(l)
"

echo "=== strace on TF import (just dlopen calls) ==="
strace -e trace=openat python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" 2>&1 | grep -i "libcu\|cudnn\|cublas\|cufft\|ENOENT.*lib" | head -30
