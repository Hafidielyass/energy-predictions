#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== CUDA Driver version ==="
nvidia-smi 2>/dev/null | grep -E "Driver Version|CUDA Version"

echo ""
echo "=== WSL libcuda version ==="
ls -la /usr/lib/wsl/lib/libcuda.so*

echo ""
echo "=== Test cudaGetDeviceCount ==="
python -c "
import ctypes

# Load cuda driver
cuda = ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1')

# Initialize
ret = cuda.cuInit(0)
print('cuInit ret:', ret, '(0=success)')

# Get device count
count = ctypes.c_int(0)
ret = cuda.cuDeviceGetCount(ctypes.byref(count))
print('cuDeviceGetCount ret:', ret, 'count:', count.value)

# Get device name
name = ctypes.create_string_buffer(256)
ret = cuda.cuDeviceGetName(name, 256, 0)
print('Device name:', name.value.decode())
" 2>&1

echo ""
echo "=== TF GPU detail with verbose logging ==="
TF_CPP_VMODULE="gpu_init=5,gpu_device=5" python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1 | grep -E "gpu_device|gpu_init|GPU|cuda|Cannot" | head -30
