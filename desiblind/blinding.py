"""Compatibility wrapper for data-vector blinding.

Prefer importing from ``desiblind.data_vector``. This module remains available
so existing notebooks, scripts, and downstream code keep working.
"""

from .data_vector import *  # noqa: F401,F403
