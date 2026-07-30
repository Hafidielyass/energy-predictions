"""
device_config.py — TensorFlow Device & Threading Configuration
===============================================================

Why this file exists
--------------------
TensorFlow >= 2.11 removed native GPU support on Windows in the
standard pip package.  This produces the noisy warning:

    WARNING:tensorflow: TensorFlow GPU support is not available on
    native Windows for TensorFlow >= 2.11. ...

This module:
  1. Suppresses that warning (and other unnecessary TF noise) cleanly
     by setting all required environment variables BEFORE tensorflow
     is imported anywhere in the process.
  2. Auto-detects and configures the best available device.
  3. Tunes CPU-thread counts for optimal performance.
  4. Provides a clear status report so users always know what device
     is actually being used.

Windows GPU upgrade paths (choose one)
---------------------------------------
A. WSL2 + CUDA  (NVIDIA only — best performance)
   - Install WSL2, Ubuntu, CUDA Toolkit, cuDNN
   - Run the project inside WSL2: `python` → uses CUDA TF
   - See: https://www.tensorflow.org/install/pip#windows-wsl2

B. tensorflow-directml-plugin  (any DirectX 12 GPU: AMD / Intel / NVIDIA)
   - Requires Python <= 3.10
   - pip install tensorflow==2.10 tensorflow-directml-plugin
   - See: https://github.com/microsoft/tensorflow-directml-plugin

C. PyTorch alternative  (fully supported on Windows + CUDA/DirectML)
   - pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   - Refactor GRU using torch.nn.GRU (API is very similar)

For this project (CPU only on Python 3.13): options A/B require a
Python version change or WSL2. The current setup is fully functional
on CPU; training takes ~2–4 h on a 20-core machine (see benchmarks).

Usage
-----
>>> # Always call BEFORE importing tensorflow anywhere
>>> from src.utils.device_config import configure_tf
>>> tf = configure_tf()          # returns the imported tensorflow module
>>> # Continue with tf.keras, tf.data, etc.
"""

import logging
import multiprocessing
import os
import sys
import warnings


def _suppress_tf_noise() -> None:
    """
    Set all environment variables that silence TF/oneDNN messages.
    Must be called BEFORE ``import tensorflow``.
    """
    # C++ level: 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Suppress oneDNN "custom operations are on" info lines
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # Suppress the Windows GPU warning specifically
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Suppress Python-level deprecation / future warnings from TF
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*TensorFlow GPU support.*")
    warnings.filterwarnings("ignore", message=".*native Windows.*")
    warnings.filterwarnings("ignore", message=".*oneDNN.*")
    warnings.filterwarnings("ignore", message=".*CUDA.*")

    # Silence the Python ``tensorflow`` logger before the module loads
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    # Absorb any remaining absl log output
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass


def _optimal_thread_count() -> int:
    """
    Return the recommended number of threads for TF operations.
    Uses all physical CPU cores (leaves 1 for the OS / main thread).
    """
    logical_cores = multiprocessing.cpu_count()
    return max(1, logical_cores)


def configure_tf(seed: int = 42, verbose: bool = True):
    """
    Configure TensorFlow for the current environment and return the
    imported ``tensorflow`` module.

    Parameters
    ----------
    seed    : int   Global random seed for reproducibility.
    verbose : bool  If True, print a one-line device status summary.

    Returns
    -------
    tensorflow module (already imported with all settings applied)

    Example
    -------
    >>> from src.utils.device_config import configure_tf
    >>> tf = configure_tf()
    >>> model = tf.keras.Sequential(...)
    """
    _suppress_tf_noise()

    # TF 2.21+ on Linux/WSL2 discovers the GPU through JAX's CUDA PJRT plugin.
    # Calling jax.devices() forces the plugin to register before TF initialises.
    # CRITICAL: disable JAX's 75% VRAM pre-allocation or it starves TF of GPU
    # memory and crashes the kernel under WSL2.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.01")
    try:
        import jax
        jax.devices()  # forces jax-cuda12-plugin GPU registration
    except Exception:
        pass

    import tensorflow as tf  # noqa: import here after env vars are set

    # ── Reproducibility ───────────────────────────────────────────────────────
    tf.random.set_seed(seed)
    import numpy as np
    np.random.seed(seed)

    # ── Thread tuning ─────────────────────────────────────────────────────────
    n_threads = _optimal_thread_count()
    # intra-op: parallelism within a single TF op (e.g. matrix multiply)
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    # inter-op: parallelism across independent TF ops in the graph
    tf.config.threading.set_inter_op_parallelism_threads(max(1, n_threads // 2))

    # ── Memory growth (prevents OOM on GPU if present) ────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    device_summary = []
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                device_summary.append(f"GPU:{gpu.name}")
            except RuntimeError:
                device_summary.append(f"GPU:{gpu.name}(no_mem_growth)")
    else:
        device_summary.append(f"CPU ({n_threads} threads)")

    # ── XLA JIT (optional speed-up on CPU for repeated ops) ──────────────────
    # tf.config.optimizer.set_jit(True)  # uncomment if XLA is available

    if verbose:
        print(f"[TF {tf.__version__}]  Device: {', '.join(device_summary)}"
              f"  |  Seed: {seed}"
              f"  |  intra={n_threads}  inter={max(1, n_threads//2)}")

    return tf


def get_device_info() -> dict:
    """
    Return a dict with full device & version information.
    Safe to call after configure_tf().
    """
    _suppress_tf_noise()
    import tensorflow as tf

    gpus  = tf.config.list_physical_devices("GPU")
    cpus  = tf.config.list_physical_devices("CPU")
    dml   = [d for d in tf.config.list_physical_devices() if "DML" in d.device_type.upper()]

    return {
        "tensorflow_version":  tf.__version__,
        "keras_version":       tf.keras.__version__,
        "python_version":      sys.version.split()[0],
        "platform":            sys.platform,
        "cpu_count":           multiprocessing.cpu_count(),
        "tf_cpu_devices":      [d.name for d in cpus],
        "tf_gpu_devices":      [d.name for d in gpus],
        "tf_dml_devices":      [d.name for d in dml],
        "active_device":       "GPU" if gpus or dml else "CPU",
        "gpu_available":       bool(gpus or dml),
        "intra_op_threads":    tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads":    tf.config.threading.get_inter_op_parallelism_threads(),
        "windows_gpu_note":    (
            "TF >= 2.11 on Windows uses CPU only. "
            "For GPU: use WSL2+CUDA (NVIDIA) or Python<=3.10 + tensorflow-directml-plugin (any GPU)."
        ) if sys.platform == "win32" and not (gpus or dml) else None,
    }


def print_device_report() -> None:
    """Print a human-readable device configuration report."""
    info = get_device_info()
    w = 50
    print("=" * w)
    print("  TensorFlow Device Report")
    print("=" * w)
    print(f"  TF version   : {info['tensorflow_version']}")
    print(f"  Keras        : {info['keras_version']}")
    print(f"  Python       : {info['python_version']}")
    print(f"  Platform     : {info['platform']}")
    print(f"  CPU cores    : {info['cpu_count']}")
    print(f"  Active device: {info['active_device']}")
    print(f"  GPU devices  : {info['tf_gpu_devices'] or 'None detected'}")
    print(f"  DirectML     : {info['tf_dml_devices'] or 'Not installed'}")
    print(f"  intra-op thd : {info['intra_op_threads']}")
    print(f"  inter-op thd : {info['inter_op_threads']}")
    if info.get("windows_gpu_note"):
        print()
        print("  NOTE (Windows GPU):")
        for chunk in _wrap(info["windows_gpu_note"], w - 4):
            print(f"    {chunk}")
        print()
        print("  GPU upgrade options:")
        print("    A) WSL2 + CUDA (NVIDIA)  → best GPU performance")
        print("       https://www.tensorflow.org/install/pip#windows-wsl2")
        print("    B) Python 3.10 + tensorflow-directml-plugin (any GPU)")
        print("       pip install tensorflow==2.10 tensorflow-directml-plugin")
    print("=" * w)


def _wrap(text: str, width: int) -> list:
    """Simple word wrapper."""
    words, line, lines = text.split(), "", []
    for w_ in words:
        if len(line) + len(w_) + 1 <= width:
            line = (line + " " + w_).lstrip()
        else:
            if line:
                lines.append(line)
            line = w_
    if line:
        lines.append(line)
    return lines
