"""Shims for torchvision features that were removed in newer releases."""

from __future__ import annotations

import sys
import types
from typing import Any


def ensure_functional_tensor_shim() -> None:
    """Ensure torchvision.transforms.functional_tensor imports even on TorchVision 0.19+.

    Real-ESRGAN (and Basicsr degradations) still import the legacy module. Newer
    TorchVision releases removed it, so we register a tiny proxy module that
    forwards the required helpers to torchvision.transforms.functional.
    """

    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except Exception:
        pass

    try:
        from torchvision.transforms import functional as F
    except Exception:
        return

    shim = types.ModuleType(module_name)

    def _rgb_to_grayscale(image: Any, num_output_channels: int = 1):
        return F.rgb_to_grayscale(image, num_output_channels=num_output_channels)

    shim.rgb_to_grayscale = _rgb_to_grayscale  # type: ignore[attr-defined]

    sys.modules[module_name] = shim
