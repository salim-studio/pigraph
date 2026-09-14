"""Backward-compatible alias for the renamed package.

``pygraph`` was renamed to ``pigraph`` (see https://github.com/salim-studio/pigraph).
``import pygraph`` keeps working and re-exports everything from ``pigraph``,
but new code should ``import pigraph``.
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "The 'pygraph' package was renamed to 'pigraph'. "
    "Use 'import pigraph' instead — 'import pygraph' is a compatibility alias.",
    DeprecationWarning,
    stacklevel=2,
)

from pigraph import *  # noqa: F401,F403
from pigraph import __version__  # noqa: F401
