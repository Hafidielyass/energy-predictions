#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== TF full build info ==="
python -c "
import tensorflow as tf
bi = tf.sysconfig.get_build_info()
for k,v in bi.items():
    if any(x in k.lower() for x in ['cuda','cudnn','gpu','compute']):
        print(f'  {k}: {v}')
"

echo ""
echo "=== Check nvidia-cuda-runtime-cu12 lib path ==="
find /home/msi/miniconda3/envs/tf_gpu -path "*/nvidia/cuda_runtime/lib/*.so*" 2>/dev/null

echo ""
echo "=== What exact lib names does TF dlopen? (LD_DEBUG) ==="
LD_DEBUG=libs python -c "
import tensorflow as tf
tf.config.list_physical_devices('GPU')
" 2>&1 | grep "libcud\|libcu[a-z]" | grep "trying\|init\|found" | head -30
