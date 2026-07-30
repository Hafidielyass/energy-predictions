#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

python -c "
import sys
sys.path.insert(0, '/mnt/c/Users/sltec/Desktop/workspace/Energy Predictions')
from src.utils.device_config import configure_tf, print_device_report

tf = configure_tf(seed=42, verbose=True)
print()
print_device_report()
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All\|WARNING:tensorflow\|WARNING:jax\|DeprecationWarning"
