"""TPU-RTX: Zero-Copy RTX-Class Ray Tracing & Physical AI Rendering on Google Cloud TPU v5p.

Quickstart:
    ```python
    import tpu_rtx

    scene = tpu_rtx.compile(mj_model, shader="fused_ao_ggx", res=(128, 128), sharding="auto")
    obs = tpu_rtx.render(mjx_data, scene)  # -> dict(rgb, depth, normal, seg_id, flow)
    ```
"""

from tpu_rtx.api import (
    DEFAULT_OUTPUTS,
    CompiledScene,
    audit_hlo,
    compile,
    lower_render,
    render,
    rollout,
    roofline,
    shard_data,
    verify_pod_topology,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "CompiledScene",
    "DEFAULT_OUTPUTS",
    "audit_hlo",
    "compile",
    "lower_render",
    "render",
    "rollout",
    "roofline",
    "shard_data",
    "verify_pod_topology",
]
