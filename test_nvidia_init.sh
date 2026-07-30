#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== nvidia package __init__.py ==="
python -c "
import nvidia
print(nvidia.__file__)
import inspect
print(inspect.getsource(nvidia))
" 2>/dev/null

echo ""
echo "=== What LD_LIBRARY_PATH looks like at Python startup ==="
python -c "
import os
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'NOT SET'))
# Check if any nvidia subpackage modifies it
import nvidia.cuda_runtime
import nvidia.cublas
import nvidia.cudnn
print('After imports LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'NOT SET'))
" 2>&1

echo ""
echo "=== TF platform check ==="
python -c "
import tensorflow.python.platform.build_info as bi
print('cuda_version:', bi.build_info.get('cuda_version'))
print('cudnn_version:', bi.build_info.get('cudnn_version'))
" 2>&1 | grep -v "^I0000"
