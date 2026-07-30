#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
python -m ipykernel install --user --name tf_gpu --display-name "Python (tf_gpu GPU)"
echo "Kernel registered. Available kernels:"
jupyter kernelspec list
