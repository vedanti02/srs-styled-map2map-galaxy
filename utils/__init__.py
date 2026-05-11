try:
    from .utils import *  # noqa: F401,F403  (requires bigfile for legacy lr2sr.py)
except ImportError:
    pass