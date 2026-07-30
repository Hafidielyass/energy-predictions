#!/bin/bash
ENV=/home/msi/miniconda3/envs/tf_gpu
ENVLIB=$ENV/lib
SITE=$ENV/lib/python3.11/site-packages/nvidia

echo "Linking CUDA libraries into $ENVLIB..."

ln -sf "$SITE/cuda_runtime/lib/libcudart.so.12"   "$ENVLIB/libcudart.so.12"
ln -sf "$SITE/cublas/lib/libcublas.so.12"          "$ENVLIB/libcublas.so.12"
ln -sf "$SITE/cublas/lib/libcublasLt.so.12"        "$ENVLIB/libcublasLt.so.12"
ln -sf "$SITE/cufft/lib/libcufft.so.11"            "$ENVLIB/libcufft.so.11"
ln -sf "$SITE/cudnn/lib/libcudnn.so.9"             "$ENVLIB/libcudnn.so.9"
ln -sf "$SITE/cudnn/lib/libcudnn_ops.so.9"         "$ENVLIB/libcudnn_ops.so.9"
ln -sf "$SITE/cudnn/lib/libcudnn_cnn.so.9"         "$ENVLIB/libcudnn_cnn.so.9"
ln -sf "$SITE/cudnn/lib/libcudnn_graph.so.9"       "$ENVLIB/libcudnn_graph.so.9"

# cusolver, cusparse, nccl
for f in "$SITE/cusolver/lib/"libcusolver.so.* "$SITE/cusparse/lib/"libcusparse.so.* "$SITE/nccl/lib/"libnccl.so.*; do
    [ -f "$f" ] && ln -sf "$f" "$ENVLIB/$(basename $f)"
done

echo "Done. Created symlinks:"
ls "$ENVLIB"/libcuda* "$ENVLIB"/libcudnn* "$ENVLIB"/libnccl* 2>/dev/null
