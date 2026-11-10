# Minimal shim so Metric3D's inference path imports cleanly without the full
# compiled mmcv package (which needs a slow from-source build and CUDA ops
# we don't need for a single-image CPU forward pass). Only implements the
# handful of symbols Metric3D's inference path actually touches.
__version__ = "2.0.0-shim"
