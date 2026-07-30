#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== TF GPU plugin .so files ==="
find /home/msi/miniconda3/envs/tf_gpu -name "*.so*" 2>/dev/null | xargs grep -l "xla_gpu\|cuda_plugin\|gpu_plugin" 2>/dev/null | head -10

echo "=== tensorflow-io-gcs-filesystem or similar GPU packages ==="
pip show tensorflow-gpu tensorflow[and-cuda] 2>/dev/null | grep -E "Name|Version"

echo "=== Check if libtensorflow_cc links to CUDA ==="
ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/tensorflow/libtensorflow_cc.so.2 2>&1 | grep -i "cu\|not found" | head -20

echo "=== What GPU plugin TF looks for ==="
python -c "
import tensorflow as tf
# TF 2.21 uses plugin mechanism
import os
tf_dir = os.path.dirname(tf.__file__)
print('TF dir:', tf_dir)
# Look for plugin_config or similar
import glob
plugins = glob.glob(tf_dir + '/*plugin*') + glob.glob(tf_dir + '/**/*plugin*', recursive=False)
print('Plugins found:', plugins[:10])
"
