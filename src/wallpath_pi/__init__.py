"""WallPath-PI package.

Recommended (canonical) API path. The training pipeline and scripts import
only from these subpackages:

- ``wallpath_pi.baselines`` (``propagation``, ``idw``) -- FSPL, log-distance, multi-wall, IDW
- ``wallpath_pi.data`` (``dataset``, ``splits``, ``sparse``, ``synthetic``)
- ``wallpath_pi.geometry`` (``raster``, ``features``, ``cache``) -- raster features
- ``wallpath_pi.evaluation`` (``metrics``)
- ``wallpath_pi.models`` (``registry``)
- ``wallpath_pi.training`` (``pipeline``) -- ``run_experiment`` / ``evaluate_saved_run``
- ``wallpath_pi.utils`` (config, paths, hashing, plotting, run summary)

The ``wallpath_pi.features``, ``wallpath_pi.physics``, and
``wallpath_pi.experiments`` packages are non-canonical compatibility modules
kept only for backward compatibility; do not import them in new code.
"""
