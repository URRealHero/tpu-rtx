"""TPU-native zero-copy MJX batch visual renderer (Stage 1, V2 SoA/Pallas, Meshes & 3DGS)."""

from tpu_mjx_render.camera import compute_camera_rays
from tpu_mjx_render.gs3d import Rigid3DGSSpec, mesh_scene_to_rigid_3dgs, render_gs3d_batch
from tpu_mjx_render.mesh_ray import MeshAssetSpec, build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.pallas_v2 import pallas_raycast_batch_v2, raycast_batch_soa
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
)
from tpu_mjx_render.renderer import (
    GBuffer,
    RenderSceneSpec,
    render_batch,
    render_exact_gbuffer,
)

__all__ = [
    "GEOM_BOX",
    "GEOM_CAPSULE",
    "GEOM_CYLINDER",
    "GEOM_ELLIPSOID",
    "GEOM_PLANE",
    "GEOM_SPHERE",
    "GBuffer",
    "RenderSceneSpec",
    "MeshAssetSpec",
    "Rigid3DGSSpec",
    "build_mesh_scene_spec",
    "compute_camera_rays",
    "mesh_scene_to_rigid_3dgs",
    "pallas_raycast_batch_v2",
    "raycast_batch_soa",
    "render_batch",
    "render_exact_gbuffer",
    "render_gs3d_batch",
    "render_mesh_batch",
]
