"""Wave 4 Scenes:
1. G=2,048 & G=4,096 Dynamic Multi-Robot Assembly & Parcel-Sorting Factory
   (4 articulated manipulators + dual conveyors + rotating carousel + overhead gantry + dense moving parcels/bins).
2. Panda GelSight / TacSL Visuo-Tactile Grasping Scene
   (head_camera + wrist_camera + dual orthographic fingertip tactile cameras grasping cube / cylinder / sphere).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np


def _add_factory_arm_xml(
    lines: List[str],
    prefix: str,
    base_x: float,
    base_y: float,
    yaw_rad: float,
    accent_rgba: str,
    with_wrist_cam: bool = False,
) -> int:
  """Add a 4-DOF articulated industrial manipulator (9 visual geoms) and return geom count."""
  c = float(np.cos(0.5 * yaw_rad))
  s = float(np.sin(0.5 * yaw_rad))
  lines.append(f'<body name="{prefix}_base" pos="{base_x:.3f} {base_y:.3f} 0.0" quat="{c:.6f} 0 0 {s:.6f}">')
  lines.append(f'  <geom name="{prefix}_ped" type="cylinder" pos="0 0 0.14" size="0.11 0.14" rgba="0.20 0.22 0.26 1" contype="0" conaffinity="0"/>')
  lines.append(f'  <geom name="{prefix}_turret" type="cylinder" pos="0 0 0.31" size="0.085 0.06" rgba="{accent_rgba}" contype="0" conaffinity="0"/>')
  lines.append(f'  <body name="{prefix}_j1" pos="0 0 0.37">')
  lines.append('    <inertial mass="2.5" pos="0 0 0.06" diaginertia="0.015 0.015 0.015"/>')
  lines.append(f'    <joint name="{prefix}_j1" type="hinge" axis="0 0 1" range="-2.6 2.6" damping="22"/>')
  lines.append(f'    <geom name="{prefix}_link1" type="capsule" fromto="0 0 0 0 0 0.18" size="0.065" rgba="0.93 0.94 0.96 1" contype="0" conaffinity="0"/>')
  lines.append(f'    <geom name="{prefix}_sh_sph" type="sphere" pos="0 0 0.18" size="0.072" rgba="{accent_rgba}" contype="0" conaffinity="0"/>')
  lines.append(f'    <body name="{prefix}_j2" pos="0 0 0.18">')
  lines.append('      <inertial mass="2.0" pos="0.18 0 0.06" diaginertia="0.012 0.012 0.012"/>')
  lines.append(f'      <joint name="{prefix}_j2" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="20"/>')
  lines.append(f'      <geom name="{prefix}_link2" type="capsule" fromto="0 0 0 0.34 0 0.10" size="0.054" rgba="0.93 0.94 0.96 1" contype="0" conaffinity="0"/>')
  lines.append(f'      <geom name="{prefix}_el_cyl" type="cylinder" pos="0.34 0 0.10" euler="1.5708 0 0" size="0.056 0.045" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>')
  lines.append(f'      <body name="{prefix}_j3" pos="0.34 0 0.10">')
  lines.append('        <inertial mass="1.5" pos="0.15 0 -0.06" diaginertia="0.008 0.008 0.008"/>')
  lines.append(f'        <joint name="{prefix}_j3" type="hinge" axis="0 1 0" range="-1.6 1.6" damping="16"/>')
  lines.append(f'        <geom name="{prefix}_link3" type="capsule" fromto="0 0 0 0.28 0 -0.14" size="0.045" rgba="0.86 0.89 0.92 1" contype="0" conaffinity="0"/>')
  lines.append(f'        <body name="{prefix}_wrist" pos="0.28 0 -0.14">')
  lines.append('          <inertial mass="0.8" pos="0.02 0 -0.04" diaginertia="0.003 0.003 0.003"/>')
  lines.append(f'          <joint name="{prefix}_j4" type="hinge" axis="0 0 1" range="-3.0 3.0" damping="6"/>')
  lines.append(f'          <geom name="{prefix}_palm" type="box" pos="0.02 0 -0.03" size="0.030 0.082 0.024" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>')
  lines.append(f'          <geom name="{prefix}_fing_l" type="box" pos="0.02 -0.052 -0.082" size="0.012 0.010 0.034" rgba="{accent_rgba}" contype="0" conaffinity="0"/>')
  if with_wrist_cam:
    lines.append(f'          <camera name="{prefix}_wrist_cam" fovy="58" pos="-0.05 0.0 0.02" xyaxes="0 1 0 0.78 0 -0.625"/>')
  lines.append('        </body>')
  lines.append('      </body>')
  lines.append('    </body>')
  lines.append('  </body>')
  lines.append('</body>')
  return 9


def make_multi_robot_factory_xml(n_target_prims: int = 4096) -> str:
  """Build an exact G=n_target_prims Dynamic Multi-Robot Assembly & Parcel-Sorting Factory MJCF."""
  palette = [
      "0.92 0.24 0.16 1", "0.14 0.56 0.92 1", "0.96 0.74 0.14 1",
      "0.16 0.72 0.42 1", "0.94 0.46 0.12 1", "0.66 0.32 0.80 1",
      "0.12 0.74 0.76 1", "0.88 0.86 0.82 1", "0.74 0.56 0.34 1",
  ]
  lines: List[str] = [
      f'<mujoco model="multi_robot_factory_G{n_target_prims}">',
      '  <compiler angle="radian" autolimits="true"/>',
      '  <statistic center="0.0 0.0 0.45" extent="3.4"/>',
      '  <option timestep="0.005" iterations="3" ls_iterations="4" integrator="implicitfast">',
      '    <flag eulerdamp="disable"/>',
      '  </option>',
      '  <visual>',
      '    <headlight diffuse="0.40 0.42 0.46" ambient="0.22 0.23 0.25" specular="0.18 0.18 0.18"/>',
      '  </visual>',
      '  <asset>',
      '    <texture type="skybox" builtin="gradient" rgb1="0.26 0.42 0.62" rgb2="0.05 0.07 0.10" width="128" height="128"/>',
      '    <texture type="2d" name="floor_tex" builtin="checker" mark="edge" rgb1="0.24 0.28 0.33" rgb2="0.17 0.20 0.24" markrgb="0.52 0.56 0.62" width="128" height="128"/>',
      '    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="8 8" reflectance="0.18"/>',
      '  </asset>',
      '  <worldbody>',
      '    <light name="factory_key" pos="1.8 -2.2 4.2" dir="-0.32 0.40 -0.86" directional="false" castshadow="true" diffuse="0.94 0.91 0.86" specular="0.32 0.32 0.32"/>',
      '    <camera name="factory_overview" pos="2.35 -2.75 1.88" xyaxes="0.76 0.65 0.0 -0.29 0.34 0.895" fovy="48"/>',
      '    <camera name="factory_closeup" pos="1.15 -1.35 1.08" xyaxes="0.76 0.65 0.0 -0.28 0.33 0.90" fovy="44"/>',
      '    <camera name="gantry_head" pos="0.0 -0.02 2.25" xyaxes="1.0 0.0 0.0 0.0 1.0 0.0" fovy="54"/>',
      '    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat" contype="1" conaffinity="1"/>',
      '    <geom name="workbench_top" type="box" pos="0.0 0.0 0.34" size="0.58 0.32 0.025" rgba="0.54 0.46 0.36 1" contype="1" conaffinity="1"/>',
      '    <geom name="conv_n_frame" type="box" pos="0.0 0.56 0.28" size="1.72 0.24 0.035" rgba="0.18 0.21 0.25 1" contype="0" conaffinity="0"/>',
      '    <geom name="conv_s_frame" type="box" pos="0.0 -0.56 0.28" size="1.72 0.24 0.035" rgba="0.18 0.21 0.25 1" contype="0" conaffinity="0"/>',
      '    <geom name="conv_n_rail1" type="box" pos="0.0 0.81 0.34" size="1.72 0.015 0.045" rgba="0.88 0.56 0.14 1" contype="0" conaffinity="0"/>',
      '    <geom name="conv_n_rail2" type="box" pos="0.0 0.31 0.34" size="1.72 0.015 0.045" rgba="0.88 0.56 0.14 1" contype="0" conaffinity="0"/>',
      '    <geom name="conv_s_rail1" type="box" pos="0.0 -0.31 0.34" size="1.72 0.015 0.045" rgba="0.88 0.56 0.14 1" contype="0" conaffinity="0"/>',
      '    <geom name="conv_s_rail2" type="box" pos="0.0 -0.81 0.34" size="1.72 0.015 0.045" rgba="0.88 0.56 0.14 1" contype="0" conaffinity="0"/>',
      '    <geom name="gantry_col_nw" type="cylinder" pos="-1.55 1.15 0.85" size="0.045 0.85" rgba="0.22 0.25 0.29 1" contype="0" conaffinity="0"/>',
      '    <geom name="gantry_col_ne" type="cylinder" pos="1.55 1.15 0.85" size="0.045 0.85" rgba="0.22 0.25 0.29 1" contype="0" conaffinity="0"/>',
      '    <geom name="gantry_col_sw" type="cylinder" pos="-1.55 -1.15 0.85" size="0.045 0.85" rgba="0.22 0.25 0.29 1" contype="0" conaffinity="0"/>',
      '    <geom name="gantry_col_se" type="cylinder" pos="1.55 -1.15 0.85" size="0.045 0.85" rgba="0.22 0.25 0.29 1" contype="0" conaffinity="0"/>',
  ]
  prim_count = 12

  # 4 Articulated Manipulators around the dual conveyor & central transfer table
  arm_specs = [
      ("arm_nw", -0.68, 0.98, -1.25, "0.94 0.42 0.12 1", True),
      ("arm_ne", 0.68, 0.98, -1.89, "0.16 0.58 0.92 1", False),
      ("arm_sw", -0.68, -0.98, 1.25, "0.94 0.42 0.12 1", False),
      ("arm_se", 0.68, -0.98, 1.89, "0.16 0.58 0.92 1", False),
  ]
  for prefix, bx, by, yaw, acc, wcam in arm_specs:
    prim_count += _add_factory_arm_xml(lines, prefix, bx, by, yaw, acc, with_wrist_cam=wcam)

  # 6 Active freejoint parcels/bins on the central transfer workbench
  free_objs = [
      ("parcel_red", "box", "0.036 0.036 0.032", "-0.22 -0.06 0.41", palette[0]),
      ("parcel_blue", "box", "0.034 0.034 0.034", "0.20 0.05 0.41", palette[1]),
      ("drum_yellow", "capsule", "0.028 0.022 0", "-0.04 0.08 0.41", palette[2]),
      ("canister_green", "capsule", "0.024 0.040 0", "0.10 -0.10 0.42", palette[3]),
      ("sphere_orange", "sphere", "0.032 0 0", "-0.14 0.10 0.41", palette[4]),
      ("tote_plum", "box", "0.044 0.034 0.026", "0.02 -0.02 0.40", palette[5]),
  ]
  for oname, otype, osize, opos, orgba in free_objs:
    lines.append(f'    <body name="{oname}" pos="{opos}">')
    lines.append('      <freejoint/>')
    lines.append('      <inertial mass="0.15" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>')
    lines.append(f'      <geom name="{oname}_geom" type="{otype}" size="{osize}" rgba="{orgba}" contype="1" conaffinity="1"/>')
    lines.append('    </body>')
    prim_count += 1

  # Moving Kinematic Trains (Conveyors + Carousel + Gantry + Side Racks)
  # Each train has its own MJX joint & actuator so all parcels/bins move dynamically!
  n_remaining = max(0, n_target_prims - prim_count)
  trains = [
      ("conv_n_train_0", "slide", "1 0 0", -0.85, 0.56, 0.32),
      ("conv_n_train_1", "slide", "1 0 0", 0.0, 0.56, 0.32),
      ("conv_n_train_2", "slide", "1 0 0", 0.85, 0.56, 0.32),
      ("conv_s_train_0", "slide", "1 0 0", -0.85, -0.56, 0.32),
      ("conv_s_train_1", "slide", "1 0 0", 0.0, -0.56, 0.32),
      ("conv_s_train_2", "slide", "1 0 0", 0.85, -0.56, 0.32),
      ("carousel_train", "hinge", "0 0 1", 0.0, 0.0, 0.365),
      ("gantry_train", "slide", "1 0 0", 0.0, 0.0, 1.48),
      ("rack_west_train", "slide", "0 1 0", -1.48, 0.0, 0.0),
      ("rack_east_train", "slide", "0 1 0", 1.48, 0.0, 0.0),
  ]
  n_trains = len(trains)
  per_train = [n_remaining // n_trains] * n_trains
  for i in range(n_remaining % n_trains):
    per_train[i] += 1

  idx_global = 0
  for t_idx, (tname, jtype, jaxis, tx, ty, tz) in enumerate(trains):
    cnt = per_train[t_idx]
    lines.append(f'    <body name="{tname}" pos="{tx:.3f} {ty:.3f} {tz:.3f}">')
    lines.append('      <inertial mass="3.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>')
    lines.append(f'      <joint name="{tname}_j" type="{jtype}" axis="{jaxis}" damping="12"/>')

    # Distribute primitives neatly along conveyors, carousel, gantry, and racks
    for k in range(cnt):
      rgba = palette[idx_global % len(palette)]
      t_sel = idx_global % 5
      if tname.startswith("conv_"):
        # Multi-row/layer parcels & tote bins on moving conveyor decks
        cols = 12
        rows = 6
        c_i = k % cols
        r_i = (k // cols) % rows
        l_i = k // (cols * rows)
        gx = -0.38 + 0.76 * ((c_i + 0.5) / cols)
        gy = -0.19 + 0.38 * ((r_i + 0.5) / rows)
        gz = 0.022 + 0.042 * l_i
        yaw = 0.31 * k
        sc = 0.55 if n_target_prims >= 3000 else 0.72
      elif tname == "carousel_train":
        ring = 0.14 + 0.12 * ((k // 16) % 3)
        ang = 2.0 * math.pi * (k % 16) / 16.0 + 0.2 * (k // 16)
        gx = ring * math.cos(ang)
        gy = 0.55 * ring * math.sin(ang)
        gz = 0.025 + 0.040 * (k // 48)
        yaw = ang
        sc = 0.50
      elif tname == "gantry_train":
        # Overhead gantry carrier rail + hanging sensor pods/parcels
        gx = -1.20 + 2.40 * ((k % 40) / 39.0)
        gy = -0.95 + 1.90 * (((k // 40) % 5) / 4.0)
        gz = -0.05 * (k // 200)
        yaw = 0.25 * k
        sc = 0.58
      else:
        # Side storage racks (west/east)
        shelf_lvl = k % 4
        along = (k // 4) % 24
        depth_row = k // (4 * 24)
        gx = -0.14 + 0.14 * (depth_row % 3)
        gy = -0.95 + 1.90 * (along / 23.0)
        gz = 0.18 + 0.34 * shelf_lvl + 0.045 * (depth_row // 3)
        yaw = 0.42 * k
        sc = 0.70

      if t_sel == 0:
        gtype, gsize = "box", f"{0.025*sc:.4f} {0.020*sc:.4f} {0.022*sc:.4f}"
      elif t_sel == 1:
        gtype, gsize = "cylinder", f"{0.019*sc:.4f} {0.026*sc:.4f}"
      elif t_sel == 2:
        gtype, gsize = "capsule", f"{0.015*sc:.4f} {0.022*sc:.4f}"
      elif t_sel == 3:
        gtype, gsize = "sphere", f"{0.021*sc:.4f}"
      else:
        gtype, gsize = "ellipsoid", f"{0.023*sc:.4f} {0.018*sc:.4f} {0.025*sc:.4f}"

      lines.append(
          f'      <geom name="fprim_{idx_global}" type="{gtype}" pos="{gx:.4f} {gy:.4f} {gz:.4f}" '
          f'euler="0 0 {yaw:.3f}" size="{gsize}" rgba="{rgba}" contype="0" conaffinity="0"/>'
      )
      idx_global += 1
    lines.append('    </body>')

  lines.append('  </worldbody>')
  lines.append('  <actuator>')
  for prefix, _, _, _, _, _ in arm_specs:
    for j in ("j1", "j2", "j3", "j4"):
      lines.append(f'    <position name="{prefix}_{j}_act" joint="{prefix}_{j}" kp="90" ctrlrange="-2.0 2.0"/>')
  for tname, jtype, _, _, _, _ in trains:
    cr = "-0.25 0.25" if jtype == "slide" else "-0.50 0.50"
    lines.append(f'    <position name="{tname}_act" joint="{tname}_j" kp="120" ctrlrange="{cr}"/>')
  lines.append('  </actuator>')
  lines.append('</mujoco>')
  return "\n".join(lines)


def assign_factory_semantic(m: mujoco.MjModel, gids: Sequence[int]) -> Tuple[int, ...]:
  """1=floor/bench/frame, 2=arms_west, 3=arms_east, 4=active_parcels, 5=conveyor_prims."""
  out: List[int] = []
  for gid in gids:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
    if name.startswith(("arm_nw", "arm_sw")):
      sem = 2
    elif name.startswith(("arm_ne", "arm_se")):
      sem = 3
    elif name.startswith(("parcel_", "drum_", "canister_", "sphere_", "tote_")):
      sem = 4
    elif name.startswith("fprim_"):
      sem = 5
    else:
      sem = 1
    out.append(sem)
  return tuple(out)


# ==============================================================================
# Task B: Panda GelSight / TacSL Visuo-Tactile Grasping Scene
# ==============================================================================

def make_panda_tactile_grasp_xml(target_shape: str = "all") -> str:
  """Panda Tabletop Grasp Scene with head + wrist + dual orthographic GelSight tactile cameras.

  Includes cube, cylinder, and sphere freejoint objects so a single compiled rig can grasp
  all three shapes across the batch without recompiling.
  """
  obj_xml = """
    <body name="target_cube" pos="0.5724 0.0 0.138">
      <freejoint/>
      <inertial mass="0.08" pos="0 0 0" diaginertia="8e-5 8e-5 8e-5"/>
      <geom name="target_cube_geom" type="box" size="0.018 0.0225 0.018" euler="0.16 0.12 0.24" condim="3"
        friction="1.2 0.03 0.003" rgba="0.88 0.22 0.18 1" contype="1" conaffinity="1" solref="0.014 1"/>
    </body>
    <body name="target_cylinder" pos="0.46 -0.14 0.072">
      <freejoint/>
      <inertial mass="0.08" pos="0 0 0" diaginertia="8e-5 8e-5 8e-5"/>
      <geom name="target_cyl_geom" type="cylinder" size="0.0225 0.020" euler="0.14 0.0 0.0"
        rgba="0.16 0.56 0.94 1" contype="0" conaffinity="0"/>
      <geom name="target_cyl_col" type="capsule" size="0.0225 0.008" euler="0.14 0.0 0.0" condim="3"
        friction="1.2 0.03 0.003" rgba="0 0 0 0" contype="1" conaffinity="1" solref="0.014 1"/>
    </body>
    <body name="target_sphere" pos="0.46 0.14 0.0735">
      <freejoint/>
      <inertial mass="0.08" pos="0 0 0" diaginertia="8e-5 8e-5 8e-5"/>
      <geom name="target_sph_geom" type="sphere" size="0.0235" condim="3"
        friction="1.2 0.03 0.003" rgba="0.96 0.72 0.14 1" contype="1" conaffinity="1" solref="0.014 1"/>
    </body>"""

  qpos_ready = (
      "0.0 0.36 0.0 -2.14 0.0 2.50 0.7854 0.021 0.021 "
      "0.5724 0.0 0.138 1.0 0.0 0.0 0.0 "
      "0.46 -0.14 0.072 1.0 0.0 0.0 0.0 "
      "0.46 0.14 0.0735 1.0 0.0 0.0 0.0"
  )
  ctrl_ready = "0.0 0.36 0.0 -2.14 0.0 2.50 0.7854 0.017"

  return f"""<mujoco model="panda_tactile_{target_shape}">
  <compiler angle="radian" autolimits="true"/>
  <statistic center="0.45 0 0.25" extent="1.0"/>

  <option timestep="0.004" iterations="6" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.24 0.24 0.24" specular="0.16 0.16 0.16"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.28 0.44 0.64" rgb2="0.05 0.07 0.11" width="128" height="128"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.24 0.30 0.38" rgb2="0.15 0.19 0.25"
      markrgb="0.65 0.70 0.75" width="128" height="128"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.18"/>
    <material name="white" rgba="0.95 0.96 0.97 1"/>
    <material name="off_white" rgba="0.86 0.89 0.92 1"/>
    <material name="black" rgba="0.20 0.22 0.25 1"/>
  </asset>

  <default>
    <default class="panda">
      <material specular="0.4" shininess="0.3"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <position forcerange="-100 100"/>
      <default class="finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>
      <default class="visual_prim">
        <geom contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="pad_col">
        <geom group="3" type="box" contype="1" conaffinity="1" condim="3" friction="1.2 0.005 0.0001"/>
      </default>
    </default>
  </default>

  <worldbody>
    <light name="sun_overhead" pos="0.55 0.25 1.8" dir="-0.12 -0.12 -1" directional="false" castshadow="true"
      diffuse="0.92 0.90 0.85" specular="0.32 0.32 0.32"/>

    <camera name="head_camera" fovy="46" pos="0.18 -0.44 0.56" xyaxes="0.78 0.625 0.0 -0.35 0.44 0.825"/>
    <camera name="overview_camera" fovy="45" pos="1.05 -0.58 0.54" xyaxes="0.48 0.88 0.0 -0.38 0.21 0.90"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
    <geom name="workbench_top" type="box" pos="0.52 0.0 0.025" size="0.36 0.42 0.025" rgba="0.52 0.40 0.28 1" contype="1" conaffinity="1"/>
    <geom name="pedestal_stand" type="box" pos="0.5724 0.0 0.084" size="0.016 0.016 0.034" rgba="0.26 0.28 0.32 1" contype="1" conaffinity="1"/>

    <body name="link0" pos="0.0 0.0 0.0" childclass="panda">
      <geom name="link0_base" type="cylinder" pos="-0.03 0 0.07" size="0.095 0.07" material="off_white" class="visual_prim"/>
      <geom name="link0_pedestal" type="cylinder" pos="0 0 0.21" size="0.068 0.08" material="white" class="visual_prim"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
        <joint name="joint1" damping="40"/>
        <geom name="link1_body" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.062" material="white" class="visual_prim"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="2.0" pos="0 -0.15 0" diaginertia="0.01 0.01 0.01"/>
          <joint name="joint2" range="-1.7628 1.7628" damping="40"/>
          <geom name="link2_collar" type="cylinder" pos="0 0 0" size="0.065 0.045" material="black" class="visual_prim"/>
          <geom name="link2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.055" material="white" class="visual_prim"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
            <joint name="joint3" damping="40"/>
            <geom name="link3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.052" material="white" class="visual_prim"/>
            <geom name="link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="black" class="visual_prim"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="2.0" pos="-0.04 0.15 0" diaginertia="0.01 0.01 0.01"/>
              <joint name="joint4" range="-3.0718 -0.0698" damping="40"/>
              <geom name="link4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.050" material="white" class="visual_prim"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.5" pos="0 0 -0.1" diaginertia="0.008 0.008 0.008"/>
                <joint name="joint5" damping="2"/>
                <geom name="link5_forearm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.044" material="off_white" class="visual_prim"/>
                <geom name="link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="black" class="visual_prim"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.0" pos="0.04 0 0" diaginertia="0.005 0.005 0.005"/>
                  <joint name="joint6" range="-0.0175 3.7525" damping="2"/>
                  <geom name="link6_wrist" type="capsule" fromto="0 0 0 0.088 0 0" size="0.040" material="white" class="visual_prim"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="0.7" pos="0 0 0.05" diaginertia="0.003 0.003 0.003"/>
                    <joint name="joint7" damping="2"/>
                    <geom name="link7_flange" type="cylinder" pos="0 0 0.06" size="0.038 0.045" material="black" class="visual_prim"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="0 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="hand_box" type="box" pos="0 0 0.030" size="0.024 0.102 0.026" material="off_white" class="visual_prim"/>
                      <geom name="hand_capsule" type="capsule" size="0.036 0.055" quat="1 1 0 0" pos="0 0 0.03" material="black" contype="1" conaffinity="1" group="3"/>
                      <camera name="wrist_camera" fovy="58" pos="-0.075 0.0 0.035" xyaxes="0 1 0 0.8575 0 -0.5145"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <inertial mass="0.04" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="finger_joint1" class="finger" damping="12"/>
                        <geom name="left_finger_vis" type="box" pos="0 0.012 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="left_finger_pad" type="box" size="0.011 0.0075 0.016" pos="0 0.010 0.038" material="off_white" class="pad_col" solref="0.014 1"/>
                        <camera name="left_tactile" projection="orthographic" fovy="0.034" pos="0 0.0155 0.038" xyaxes="-1 0 0 0 0 1"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.04" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="finger_joint2" class="finger" damping="12"/>
                        <geom name="right_finger_vis" type="box" pos="0 0.012 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="right_finger_pad" type="box" size="0.011 0.0075 0.016" pos="0 0.010 0.038" material="off_white" class="pad_col" solref="0.014 1"/>
                        <camera name="right_tactile" projection="orthographic" fovy="0.034" pos="0 0.0155 0.038" xyaxes="-1 0 0 0 0 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    {obj_xml}
  </worldbody>

  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <actuator>
    <position class="panda" name="act1" joint="joint1" kp="1000" kv="20" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="act2" joint="joint2" kp="1000" kv="20" ctrlrange="-1.7628 1.7628"/>
    <position class="panda" name="act3" joint="joint3" kp="750" kv="4" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="act4" joint="joint4" kp="750" kv="4" ctrlrange="-3.0718 -0.0698"/>
    <position class="panda" name="act5" joint="joint5" kp="300" kv="2" forcerange="-12 12" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="act6" joint="joint6" kp="300" kv="2" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>
    <position class="panda" name="act7" joint="joint7" kp="300" kv="2" forcerange="-12 12"/>
    <general class="panda" name="act8" joint="finger_joint1" ctrlrange="0 0.04" gainprm="450 0 0"
      biasprm="0 -450 -10" forcerange="-100 100"/>
  </actuator>

  <keyframe>
    <key name="grasp_ready" qpos="{qpos_ready}" ctrl="{ctrl_ready}"/>
  </keyframe>
</mujoco>
"""


def assign_tactile_semantic(m: mujoco.MjModel, gids: Sequence[int]) -> Tuple[int, ...]:
  """1=floor/bench/pedestal, 2=panda_arm, 3=target_object."""
  out: List[int] = []
  for gid in gids:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
    if name in ("floor", "workbench_top", "pedestal_stand"):
      sem = 1
    elif name.startswith("target_"):
      sem = 3
    else:
      sem = 2
    out.append(sem)
  return tuple(out)
