"""TPU-RTX Ecosystem & Studio Adapters.

Provides lazy, zero-overhead module exports so importing this package on host
CPUs or in non-Blender/non-OpenUSD environments does not eagerly import heavy
optional dependencies (`jax`, `bpy`, `pxr`).
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

__all__ = [
    "mjx_universal",
    "isaac_replicator",
    "openusd_loader",
    "blend_compiler",
    "tpu_gsplat",
    "tpu_optix",
    "tpu_dlss",
    "tpu_restir",
    "ADAPTER_DESCRIPTIONS",
]

ADAPTER_DESCRIPTIONS: Dict[str, str] = {
    "mjx_universal": "Universal MuJoCo/MJX & Menagerie zero-copy Pallas raycaster + differentiable PBR.",
    "isaac_replicator": "Isaac Lab / Replicator domain randomization, TiledCamera, LiDAR, and tactile sensors.",
    "openusd_loader": "OpenUSD (.usda/.usdc/.usd) stage/instancer loader for Disney Moana & industrial worlds.",
    "blend_compiler": "Blender Cycles Shader Graph & asset compiler to Wide-BVH-64 TPU packs.",
    "tpu_gsplat": "Native TPU 3DGS / 4DGS splatting & differentiable training (host_callbacks=0).",
    "tpu_optix": "OptiX-style BVH ray query, inverse rendering & volumetric caustics on TPU.",
    "tpu_dlss": "TPU-DLSS 3.5 temporal super-resolution, MC denoising & neural relighting on MXU.",
    "tpu_restir": "TPU-ReSTIR DI spatio-temporal many-light reservoir resampling kernel.",
}


def __getattr__(name: str) -> ModuleType:
  if name in ADAPTER_DESCRIPTIONS:
    return importlib.import_module(f"{__name__}.{name}")
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(list(globals().keys()) + list(ADAPTER_DESCRIPTIONS.keys()))
