"""NumPy backend — currently the only supported backend."""

from __future__ import annotations

import numpy as np


def is_available() -> bool:
    return True


def get_array_module():
    return np
