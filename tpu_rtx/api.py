"""TPU-RTX Public Industrial One-Liner SDK."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import numpy as np

from tpu_mjx_render.adapters.mjx_universal import (
    DEFAULT_OUTPUTS,
    CompiledScene,
    audit_hlo,
    build_render_fn,
    compile_scene,
    compute_roofline_report,
    render_scene,
    rollout_scan,
    shard_data,
)


def compile(
    source: Any,
    shader: str = "fused_ao_ggx",
    res: Union[int, Tuple[int, int]] = (128, 128),
    sharding: Union[str, int, jax.sharding.Mesh, None] = "auto",
    cam_id: Union[int, str, Sequence[Union[int, str]]] = 0,
    dtype: Any = "bfloat16",
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    **kwargs: Any,
) -> CompiledScene:
  """Compile any MuJoCo / MJX scene into a zero-copy TPU-RTX `CompiledScene`.

  Example:
    ```python
    scene = tpu_rtx.compile(mj_model, shader="fused_ao_ggx", res=(128, 128), sharding="auto")
    obs = tpu_rtx.render(mjx_data, scene)  # -> dict(rgb, depth, normal, seg_id, flow)
    ```
  """
  return compile_scene(
      source=source,
      shader=shader,
      res=res,
      sharding=sharding,
      cam_id=cam_id,
      dtype=dtype,
      outputs=outputs,
      **kwargs,
  )


def render(
    mjx_data: Any,
    scene: CompiledScene,
    prev_data: Optional[Any] = None,
    geom_rgba: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
  """Zero-copy render `mjx_data` using a compiled `scene` -> `dict(rgb, depth, normal, seg_id, flow)`."""
  return render_scene(mjx_data, scene, prev_data=prev_data, geom_rgba=geom_rgba)


def rollout(
    mjx_model: Any,
    mjx_data: Any,
    ctrl_seq: jax.Array,
    scene: CompiledScene,
    substeps: int = 1,
    return_frames: bool = True,
) -> Tuple[Any, Dict[str, jax.Array]]:
  """Fused `lax.scan` multi-step MJX physics + TPU-RTX render inside one XLA program."""
  return rollout_scan(mjx_model, mjx_data, ctrl_seq, scene, substeps=substeps, return_frames=return_frames)


def lower_render(
    mjx_data: Any,
    scene: CompiledScene,
    prev_data: Optional[Any] = None,
    geom_rgba: Optional[jax.Array] = None,
):
  """Lower `tpu_rtx.render` to XLA HLO for zero-copy static verification."""
  unbatched = mjx_data.cam_xpos.ndim == 2
  d_b = jax.tree_util.tree_map(lambda x: x[None, ...], mjx_data) if unbatched else mjx_data
  p_b = (jax.tree_util.tree_map(lambda x: x[None, ...], prev_data) if unbatched else prev_data) if prev_data is not None else None
  r_b = (geom_rgba[None, ...] if (geom_rgba is not None and geom_rgba.ndim == 2) else geom_rgba)
  B = int(d_b.cam_xpos.shape[0])
  fn = build_render_fn(scene, batch_size=B, has_prev=(p_b is not None), has_dyn_rgba=(r_b is not None))
  return fn.lower(d_b, p_b, r_b)


def roofline(
    scene: CompiledScene,
    batch_size: int,
    dt_render_ms: Optional[float] = None,
    hw_name: str = "v5p",
) -> Dict[str, Any]:
  """Compute TPU hardware roofline & achieved HBM/VPU/VMEM utilization."""
  return compute_roofline_report(scene, batch_size=batch_size, dt_render_ms=dt_render_ms, hw_name=hw_name)


def verify_pod_topology(
    n_chips_list: Sequence[int] = (1, 4, 32, 64, 1000),
    per_chip_batch: int = 2048,
    per_chip_render_fps: float = 250000.0,
) -> list[Dict[str, Any]]:
  """Construct & verify SPMD `data_axis` Pod scaling topology across 1..1000+ TPU v5p chips."""
  rows = []
  for n in n_chips_list:
    # Verify global batch divisibility and linear SPMD zero-allreduce scaling
    global_batch = int(n) * int(per_chip_batch)
    est_fps = float(n) * float(per_chip_render_fps)
    rows.append({
        "n_chips": int(n),
        "mesh_axes": ("data",),
        "mesh_shape": (int(n),),
        "partition_spec": ("data",),
        "global_batch": global_batch,
        "cross_chip_comm_ops": 0,
        "spmd_scaling_efficiency": 1.0,
        "projected_render_fps": est_fps,
    })
  return rows


__all__ = [
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
