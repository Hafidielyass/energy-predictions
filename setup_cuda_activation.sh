#!/bin/bash
# Ensure the conda activation script sets LD_LIBRARY_PATH for CUDA libs
mkdir -p /home/msi/miniconda3/envs/tf_gpu/etc/conda/activate.d
mkdir -p /home/msi/miniconda3/envs/tf_gpu/etc/conda/deactivate.d

SITE=/home/msi/miniconda3/envs/tf_gpu/lib/python3.11/site-packages/nvidia

cat > /home/msi/miniconda3/envs/tf_gpu/etc/conda/activate.d/cuda_libs.sh << EOF
#!/bin/bash
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/msi/miniconda3/envs/tf_gpu/lib
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
EOF

echo "Activation script written:"
cat /home/msi/miniconda3/envs/tf_gpu/etc/conda/activate.d/cuda_libs.sh
