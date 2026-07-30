#!/bin/bash
ENV=/home/msi/miniconda3/envs/tf_gpu
ENVLIB=$ENV/lib
CUDNN=$ENV/lib/python3.11/site-packages/nvidia/cudnn/lib

VER=9.23.2

# Remove bad symlinks with double .so
rm -f $ENVLIB/libcudnn*.so.so.* $CUDNN/libcudnn*.so.so.*

echo "Creating versioned cuDNN symlinks (version $VER)..."

for so in libcudnn.so.9 libcudnn_ops.so.9 libcudnn_cnn.so.9 libcudnn_graph.so.9 \
          libcudnn_adv.so.9 libcudnn_engines_precompiled.so.9 \
          libcudnn_engines_runtime_compiled.so.9 libcudnn_engines_tensor_ir.so.9 \
          libcudnn_ext.so.9 libcudnn_heuristic.so.9; do
    base="${so%.9}"           # strips trailing .9 → e.g. libcudnn_graph.so
    src="$CUDNN/$so"
    versioned="$base.$VER"   # e.g. libcudnn_graph.so.9.23.2
    ln -sf "$src" "$CUDNN/$versioned"
    ln -sf "$src" "$ENVLIB/$versioned"
done

echo "Done. Versioned symlinks in ENVLIB:"
ls $ENVLIB/libcudnn*.so.9.*.* 2>/dev/null
