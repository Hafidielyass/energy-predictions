#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

SITE=/home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/nvidia

echo "=== SONAMEs ==="
for f in libcudnn.so.9 libcudnn_graph.so.9 libcudnn_ops.so.9 libcudnn_cnn.so.9; do
    path=$SITE/cudnn/lib/$f
    soname=$(objdump -p $path 2>/dev/null | grep SONAME | awk '{print $2}')
    echo "$f SONAME: $soname"
done

echo ""
echo "=== All DT_NEEDED for libcudnn_graph.so.9 ==="
objdump -p $SITE/cudnn/lib/libcudnn_graph.so.9 2>/dev/null | grep NEEDED

echo ""
echo "=== cudnn_graph dependencies missing? ==="
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ldd $SITE/cudnn/lib/libcudnn_graph.so.9 2>&1 | grep "not found"

echo ""
echo "=== TF internal GPU test ==="
python -c "
import tensorflow as tf
print('is_built_with_cuda:', tf.test.is_built_with_cuda())
print('is_gpu_available:', tf.test.is_gpu_available(cuda_only=False))
" 2>&1 | grep -v "^I0000\|oneDNN\|rebuild\|WARNING: All\|instructions"
