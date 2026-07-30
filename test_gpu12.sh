#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== ldd on libcudnn_graph.so.9 ==="
ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_graph.so.9 2>&1 | grep "not found\|libcu"

echo ""
echo "=== ldd on libcublas.so.12 ==="
ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12 2>&1 | grep "not found\|libcu"

echo ""
echo "=== ldd on libtensorflow_cc.so.2 ==="
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/tensorflow/libtensorflow_cc.so.2 2>&1 | grep "not found\|libcu"
