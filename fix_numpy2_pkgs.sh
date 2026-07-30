#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "=== Upgrading packages for NumPy 2.x compatibility ==="
pip install "shap>=0.46" "xgboost>=2.0" "numba>=0.59" "lightgbm>=4.3" 2>&1 | tail -15

echo ""
echo "=== Verify all imports work ==="
python -c "
import numpy as np; print('numpy:', np.__version__)
import pandas as pd; print('pandas:', pd.__version__)
import sklearn; print('sklearn:', sklearn.__version__)
import scipy; print('scipy:', scipy.__version__)
import xgboost; print('xgboost:', xgboost.__version__)
import shap; print('shap:', shap.__version__)
import matplotlib; print('matplotlib:', matplotlib.__version__)
print('All imports OK')
" 2>&1 | grep -v "WARNING\|UserWarning\|FutureWarning"
