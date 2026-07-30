#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== ldd on libtensorflow_cc.so ==="
ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/tensorflow/libtensorflow_cc.so.2 2>&1 | grep -E "cuda|not found|libcu" | head -20

echo ""
echo "=== ldd on _pywrap_tensorflow_internal.so ==="
ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so 2>&1 | grep -E "cuda|not found|libcu" | head -20

echo ""
echo "=== pip list | tensorflow-related ==="
pip list 2>/dev/null | grep -i "tensor\|keras\|cuda\|jax\|xla"

echo ""
echo "=== TF build info ==="
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())" 2>/dev/null | tr ',' '\n' | grep -i "cuda\|cudnn\|gpu"
