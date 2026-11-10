from mmengine import Config, DictAction  # noqa: F401
from mmengine.utils import get_git_hash  # noqa: F401


def collect_env():
    """Stub: real mmcv.utils.collect_env(); unused on the inference path
    (only imported, never called, in mono/utils/comm.py)."""
    return {}
