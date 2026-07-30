#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== TF .so files with unresolved deps ==="
find /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/tensorflow -name "*.so" 2>/dev/null | while read lib; do
    missing=$(ldd "$lib" 2>/dev/null | grep "not found")
    if [ -n "$missing" ]; then
        echo "MISSING in $lib:"
        echo "$missing"
    fi
done

echo ""
echo "=== Check libtensorflow_cc.so.2 full deps ==="
ldd /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/tensorflow/libtensorflow_cc.so.2 2>&1 | head -30
