#!/bin/bash
ENV=/home/msi/miniconda3/envs/tf_gpu
ENVLIB=$ENV/lib
SITE=$ENV/lib/python3.11/site-packages/nvidia

ln -sf $SITE/cuda_cupti/lib/libcupti.so.12  $ENVLIB/libcupti.so.12
ln -sf $SITE/nccl/lib/libnccl.so.2          $ENVLIB/libnccl.so.2

# cusolver
find $SITE/cusolver/lib -name "*.so*" 2>/dev/null | while read f; do
    ln -sf "$f" "$ENVLIB/$(basename $f)"
done

# cusparse
find $SITE/cusparse/lib -name "*.so*" 2>/dev/null | while read f; do
    ln -sf "$f" "$ENVLIB/$(basename $f)"
done

echo "Done:"
ls $ENVLIB/libcupti* $ENVLIB/libcusolver* $ENVLIB/libcusparse* $ENVLIB/libnccl* 2>/dev/null
