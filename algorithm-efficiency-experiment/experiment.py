#!/usr/bin/env python3
"""Entry point; set numerical-library thread limits before importing NumPy."""
import os

# A spawn worker must retain the environment selected before process creation.
# Replaying this setup as __mp_main__ would overwrite measured CPU policies.
if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

from efficiency.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
