#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# Check ALL possible libs TF might need
echo "=== Testing each CUDA lib by full path ==="
python -c "
import ctypes

# All possible candidates TF 2.21 loads
libs = {
    'libcuda.so.1':    '/usr/lib/wsl/lib/libcuda.so.1',
    'libcudart.so.12': '/home/msi/miniconda3/envs/tf_gpu/lib/libcudart.so.12',
    'libcublas.so.12': '/home/msi/miniconda3/envs/tf_gpu/lib/libcublas.so.12',
    'libcublasLt.so.12': '/home/msi/miniconda3/envs/tf_gpu/lib/libcublasLt.so.12',
    'libcufft.so.11':  '/home/msi/miniconda3/envs/tf_gpu/lib/libcufft.so.11',
    'libcudnn.so.9':   '/home/msi/miniconda3/envs/tf_gpu/lib/libcudnn.so.9',
}

# Also check cupti
import glob
site = '/home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/nvidia'
cupti = glob.glob(site + '/cuda_cupti/lib/*.so*')
print('cupti files:', cupti)

for name, path in libs.items():
    try:
        ctypes.CDLL(path)
        print(f'OK: {name}')
    except Exception as e:
        print(f'FAIL: {name} — {e}')
"

echo ""
echo "=== Check cupti symlink ==="
ls /home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/nvidia/cuda_cupti/lib/

echo ""
echo "=== Check libcublasLt ==="
ls /home/msi/miniconda3/envs/tf_gpu/lib/libcublas* 2>/dev/null
