from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True, num_threads: int | None = None) -> None:
    """Seed Python's ``random`` and NumPy's legacy global RNG for a run.

    This helper controls the randomness sources WallPath-PI actually relies on:
    the process-wide :mod:`random` generator and NumPy's legacy global RNG via
    :func:`numpy.random.seed`. Together with the SHA-derived per-sample sparse
    mask seeds (see :mod:`wallpath_pi.data.sparse`) and the explicit
    ``random_state`` values passed to every scikit-learn estimator, this makes
    anchor sampling and model fitting reproducible for a fixed configuration.

    What this helper does **not** by itself guarantee when called from an
    already-running interpreter:

    * It assigns ``PYTHONHASHSEED`` in :data:`os.environ`, but CPython fixes its
      hash randomization at interpreter start-up. Assigning the variable here
      only influences *child* processes; to pin hashing for the current process
      the variable must be exported *before* Python launches.
    * When ``num_threads`` is given it assigns ``OMP_NUM_THREADS`` /
      ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``, but BLAS and OpenMP
      back-ends read these when their thread pools are first created, which for
      many builds happens at import time. Setting them here is best-effort and
      reliably takes effect only when exported before the process starts.

    For fully pinned hashing and thread counts, export the environment before
    launching Python. PowerShell::

        $env:PYTHONHASHSEED = "0"; $env:OMP_NUM_THREADS = "1"
        $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
        python scripts/train.py --config configs/config.yaml

    POSIX shells::

        PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
            MKL_NUM_THREADS=1 python scripts/train.py --config configs/config.yaml

    The ``deterministic`` flag is accepted for call-site compatibility; a run's
    numerical determinism comes from the seeds and explicit ``random_state``
    values described above rather than from this flag.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(int(num_threads))
        os.environ["OPENBLAS_NUM_THREADS"] = str(int(num_threads))
        os.environ["MKL_NUM_THREADS"] = str(int(num_threads))
