#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

python -c "
# Intercept ctypes.CDLL to log what TF tries to open
import ctypes, ctypes.util
_orig_CDLL = ctypes.CDLL.__init__

def _patched_init(self, name, *args, **kw):
    if name and ('libcu' in str(name) or 'libnv' in str(name) or 'libtensor' in str(name)):
        try:
            _orig_CDLL(self, name, *args, **kw)
            print(f'LOADED: {name}')
        except OSError as e:
            print(f'FAILED: {name} -- {e}')
        return
    _orig_CDLL(self, name, *args, **kw)

ctypes.CDLL.__init__ = _patched_init

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
" 2>&1 | grep -v "^I0000\|oneDNN\|instructions\|rebuild\|WARNING: All"
