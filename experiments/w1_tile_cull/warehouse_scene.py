"""Wave 2: Visually Recognizable Cluttered Robotics Warehouse & Kitchen Manipulation MJCF Generator.

Builds parameterized MuJoCo scenes with 50, 200, or 1,000+ moving primitives per environment:
- Dual-arm articulated robotics workstations flanking a central packing/prep workbench
- Multi-tier warehouse shelving racks & animated conveyor/carousel trays (all primitives move via MJX joints)
- Colorful storage bins, mugs, bottles, cans, bowls, tools, and active tabletop freejoint grasp objects
"""

from __future__ import annotations

import math
from typing import List


def _add_arm_xml(lines: List[str], prefix: str, base_x: float, base_y: float, yaw: float) -> None:
  c, s = math.cos(yaw), math.sin(yaw)
  # Base pedestal + 6-DOF articulated arm
  lines.append(f'<body name="{prefix}_base" pos="{base_x:.3f} {base_y:.3f} 0.0" euler="0 0 {yaw:.4f}">')
  lines.append('  <geom type="cylinder" pos="0 0 0.08" size="0.095 0.08" rgba="0.22 0.24 0.27 1" contype="0" conaffinity="0"/>')
  lines.append('  <geom type="cylinder" pos="0 0 0.21" size="0.068 0.07" rgba="0.93 0.94 0.96 1" contype="0" conaffinity="0"/>')
  lines.append(f'  <body name="{prefix}_j1" pos="0 0 0.30">')
  lines.append(f'    <inertial mass="2.0" pos="0 0 0.05" diaginertia="0.01 0.01 0.01"/>')
  lines.append(f'    <joint name="{prefix}_j1" type="hinge" axis="0 0 1" damping="20"/>')
  lines.append('    <geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.058" rgba="0.95 0.96 0.97 1" contype="0" conaffinity="0"/>')
  lines.append('    <geom type="sphere" pos="0 0 0.14" size="0.062" rgba="0.92 0.38 0.12 1" contype="0" conaffinity="0"/>')
  lines.append(f'    <body name="{prefix}_j2" pos="0 0 0.14">')
  lines.append(f'      <inertial mass="1.8" pos="0.14 0 0" diaginertia="0.01 0.01 0.01"/>')
  lines.append(f'      <joint name="{prefix}_j2" type="hinge" axis="0 1 0" damping="20"/>')
  lines.append('      <geom type="capsule" fromto="0 0 0 0.28 0 0.08" size="0.050" rgba="0.95 0.96 0.97 1" contype="0" conaffinity="0"/>')
  lines.append('      <geom type="cylinder" pos="0.28 0 0.08" euler="1.5708 0 0" size="0.052 0.04" rgba="0.20 0.22 0.25 1" contype="0" conaffinity="0"/>')
  lines.append(f'      <body name="{prefix}_j3" pos="0.28 0 0.08">')
  lines.append(f'        <inertial mass="1.4" pos="0.12 0 -0.05" diaginertia="0.008 0.008 0.008"/>')
  lines.append(f'        <joint name="{prefix}_j3" type="hinge" axis="0 1 0" damping="15"/>')
  lines.append('        <geom type="capsule" fromto="0 0 0 0.24 0 -0.10" size="0.042" rgba="0.88 0.90 0.93 1" contype="0" conaffinity="0"/>')
  lines.append('        <geom type="sphere" pos="0.24 0 -0.10" size="0.044" rgba="0.14 0.50 0.80 1" contype="0" conaffinity="0"/>')
  lines.append(f'        <body name="{prefix}_wrist" pos="0.24 0 -0.10">')
  lines.append(f'          <inertial mass="0.8" pos="0.05 0 -0.04" diaginertia="0.003 0.003 0.003"/>')
  lines.append(f'          <joint name="{prefix}_j4" type="hinge" axis="0 0 1" damping="5"/>')
  lines.append('          <geom type="box" pos="0.04 0 -0.03" size="0.028 0.085 0.024" rgba="0.24 0.26 0.29 1" contype="0" conaffinity="0"/>')
  lines.append('          <geom type="box" pos="0.04 -0.055 -0.085" size="0.012 0.010 0.035" rgba="0.85 0.88 0.92 1" contype="0" conaffinity="0"/>')
  lines.append('          <geom type="box" pos="0.04 0.055 -0.085" size="0.012 0.010 0.035" rgba="0.85 0.88 0.92 1" contype="0" conaffinity="0"/>')
  lines.append('        </body>')
  lines.append('      </body>')
  lines.append('    </body>')
  lines.append('  </body>')
  lines.append('</body>')
  del c, s


def build_warehouse_mjcf(n_target_prims: int = 200) -> str:
  """Generate a cluttered robotics warehouse & kitchen manipulation scene with >= n_target_prims geoms."""
  palette = [
      "0.86 0.18 0.16 1",  # tomato red
      "0.15 0.52 0.82 1",  # cobalt blue
      "0.94 0.72 0.14 1",  # amber yellow
      "0.18 0.68 0.38 1",  # emerald green
      "0.92 0.44 0.14 1",  # safety orange
      "0.62 0.30 0.74 1",  # plum
      "0.14 0.70 0.72 1",  # teal
      "0.90 0.88 0.84 1",  # ceramic white
      "0.74 0.56 0.36 1",  # cardboard kraft
  ]

  lines: List[str] = [
      '<mujoco model="cluttered_warehouse">',
      '  <compiler angle="radian" autolimits="true"/>',
      '  <statistic center="0.0 0.0 0.5" extent="2.5"/>',
      '  <option timestep="0.005" iterations="3" ls_iterations="4" integrator="implicitfast">',
      '    <flag eulerdamp="disable"/>',
      '  </option>',
      '  <visual>',
      '    <headlight diffuse="0.42 0.42 0.45" ambient="0.22 0.22 0.24" specular="0.15 0.15 0.15"/>',
      '  </visual>',
      '  <asset>',
      '    <texture type="skybox" builtin="gradient" rgb1="0.34 0.52 0.72" rgb2="0.08 0.10 0.14" width="128" height="128"/>',
      '    <texture type="2d" name="floor_tex" builtin="checker" mark="edge" rgb1="0.28 0.32 0.36" rgb2="0.20 0.24 0.28" markrgb="0.55 0.60 0.65" width="128" height="128"/>',
      '    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="6 6" reflectance="0.15"/>',
      '  </asset>',
      '  <worldbody>',
      '    <light name="warehouse_key" pos="1.4 -1.6 3.6" dir="-0.32 0.38 -0.88" directional="false" castshadow="true" diffuse="0.92 0.90 0.85" specular="0.30 0.30 0.30"/>',
      '    <camera name="overview" pos="1.85 -2.25 1.55" xyaxes="0.77 0.64 0.0 -0.28 0.34 0.90" fovy="48"/>',
      '    <camera name="closeup" pos="0.95 -1.15 0.95" xyaxes="0.77 0.64 0.0 -0.28 0.34 0.90" fovy="44"/>',
      '    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat" contype="1" conaffinity="1"/>',
  ]
  prim_count = 1  # floor

  # Central workbench + kitchen/warehouse counter
  lines.append('    <geom name="bench_top" type="box" pos="0.0 0.0 0.38" size="0.72 0.44 0.025" rgba="0.78 0.68 0.54 1" contype="1" conaffinity="1"/>')
  lines.append('    <geom name="bench_leg1" type="box" pos="-0.64 -0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>')
  lines.append('    <geom name="bench_leg2" type="box" pos="0.64 -0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>')
  lines.append('    <geom name="bench_leg3" type="box" pos="-0.64 0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>')
  lines.append('    <geom name="bench_leg4" type="box" pos="0.64 0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>')
  prim_count += 5

  # Dual articulated robot workstations
  _add_arm_xml(lines, "left_arm", -0.58, -0.55, 0.95)
  _add_arm_xml(lines, "right_arm", 0.58, -0.55, 2.20)
  prim_count += 18  # 9 geoms per arm

  # Rear & side multi-tier warehouse shelving racks (3 racks x (4 posts + 3 shelves = 7 geoms) = 21 geoms)
  rack_specs = [
      ("rack_back", 0.0, 0.98, 0.0),
      ("rack_left", -1.25, 0.25, 1.35),
      ("rack_right", 1.25, 0.25, -1.35),
  ]
  for rname, rx, ry, ryaw in rack_specs:
    lines.append(f'    <body name="{rname}" pos="{rx:.3f} {ry:.3f} 0.0" euler="0 0 {ryaw:.4f}">')
    for sx in (-0.65, 0.65):
      for sy in (-0.22, 0.22):
        lines.append(f'      <geom type="cylinder" pos="{sx:.2f} {sy:.2f} 0.62" size="0.022 0.62" rgba="0.28 0.30 0.34 1" contype="0" conaffinity="0"/>')
        prim_count += 1
    for sz in (0.18, 0.58, 0.98):
      lines.append(f'      <geom type="box" pos="0 0 {sz:.2f}" size="0.68 0.24 0.018" rgba="0.45 0.48 0.52 1" contype="0" conaffinity="0"/>')
      prim_count += 1
    lines.append('    </body>')

  # Active tabletop freejoint objects (6 objects colliding with bench_top)
  active_objs = [
      ("cube_red", "box", "0.032 0.032 0.032", "-0.18 -0.08 0.44", palette[0]),
      ("cube_blue", "box", "0.030 0.030 0.035", "0.14 -0.10 0.44", palette[1]),
      ("can_yellow", "cylinder", "0.028 0.045 0", "-0.02 0.06 0.45", palette[2]),
      ("bottle_green", "capsule", "0.024 0.050 0", "0.28 0.05 0.48", palette[3]),
      ("orange_ball", "sphere", "0.032 0 0", "-0.30 0.08 0.44", palette[4]),
      ("bowl_plum", "ellipsoid", "0.045 0.045 0.025", "0.04 -0.16 0.43", palette[5]),
  ]
  for oname, otype, osize, opos, orgba in active_objs:
    lines.append(f'    <body name="{oname}" pos="{opos}">')
    lines.append('      <freejoint/>')
    lines.append('      <inertial mass="0.12" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>')
    lines.append(f'      <geom type="{otype}" size="{osize}" rgba="{orgba}" contype="1" conaffinity="1"/>')
    lines.append('    </body>')
    prim_count += 1

  # Animated dynamic tray/carousel groups holding all remaining cluttered warehouse/kitchen props!
  # Using 4 moving articulated carrier bodies so all props dynamically translate/rotate in MJX!
  n_remaining = max(0, n_target_prims - prim_count)
  # Divide props across 4 animated zones: bench_tray, rack_back_slider, rack_left_rot, rack_right_rot
  zones = [
      ("zone_bench", "slide", "1 0 0", 0.0, 0.05, 0.405),
      ("zone_back", "slide", "1 0 0", 0.0, 0.98, 0.0),
      ("zone_left", "hinge", "0 0 1", -1.25, 0.25, 0.0),
      ("zone_right", "hinge", "0 0 1", 1.25, 0.25, 0.0),
  ]

  per_zone = [n_remaining // 4] * 4
  for i in range(n_remaining % 4):
    per_zone[i] += 1

  idx_global = 0
  for z_idx, (zname, jtype, jaxis, zx, zy, zz) in enumerate(zones):
    count_z = per_zone[z_idx]
    lines.append(f'    <body name="{zname}" pos="{zx:.3f} {zy:.3f} {zz:.3f}">')
    lines.append('      <inertial mass="1.0" pos="0 0 0" diaginertia="0.05 0.05 0.05"/>')
    lines.append(f'      <joint name="{zname}_j" type="{jtype}" axis="{jaxis}" damping="2"/>')

    for k in range(count_z):
      col = palette[idx_global % len(palette)]
      # Deterministic jittered grid placement per zone
      h1 = ((idx_global * 2654435761) & 0xFFFF) / 65535.0
      h2 = (((idx_global + 17) * 2246822519) & 0xFFFF) / 65535.0
      h3 = (((idx_global + 53) * 3266489917) & 0xFFFF) / 65535.0

      if z_idx == 0:
        # Bench items: mugs, bottles, bins, tools arranged on tabletop
        cols = max(4, int(math.ceil(math.sqrt(max(count_z, 1) * 1.6))))
        row = k // cols
        col_i = k % cols
        rows = max(1, (count_z + cols - 1) // cols)
        px = -0.60 + (col_i + 0.5) * (1.20 / cols) + (h1 - 0.5) * 0.02
        py = -0.34 + (row + 0.5) * (0.64 / rows) + (h2 - 0.5) * 0.02
        pz = 0.025 + (k % 2) * 0.012
        rot = f'euler="0 0 {h3 * 3.14:.3f}"'
      else:
        # Multi-tier shelving bins, bottles, crates, cans, bowls across 3 shelf levels (0.22, 0.62, 1.02)
        tier = k % 3
        k_tier = k // 3
        n_tier = max(1, (count_z + 2) // 3)
        cols = max(4, int(math.ceil(math.sqrt(n_tier * 2.5))))
        rows = max(1, (n_tier + cols - 1) // cols)
        r_i = k_tier // cols
        c_i = k_tier % cols
        lx = -0.56 + (c_i + 0.5) * (1.12 / cols) + (h1 - 0.5) * 0.015
        ly = -0.16 + (r_i + 0.5) * (0.32 / rows) + (h2 - 0.5) * 0.015
        if z_idx == 2:
          px, py = ly * 0.9, lx * 0.9
        elif z_idx == 3:
          px, py = -ly * 0.9, -lx * 0.9
        else:
          px, py = lx, ly
        pz = 0.22 + tier * 0.40 + 0.03
        rot = f'euler="0 0 {h3 * 3.14:.3f}"'

      # Alternate recognizable primitive shapes: crate/bin (box), bottle/mug (cylinder), tool/handle (capsule), fruit/jar (sphere), bowl (ellipsoid)
      kind = idx_global % 5
      scale = 0.55 if n_target_prims >= 500 else (0.78 if n_target_prims >= 150 else 1.0)
      if kind == 0:  # bin / parcel box
        sx, sy, sz = (0.030 + 0.02 * h1) * scale, (0.025 + 0.015 * h2) * scale, (0.026 + 0.02 * h3) * scale
        lines.append(f'      <geom type="box" pos="{px:.3f} {py:.3f} {pz:.3f}" {rot} size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="{col}" contype="0" conaffinity="0"/>')
      elif kind == 1:  # bottle / mug cylinder
        sr, sh = (0.018 + 0.010 * h1) * scale, (0.032 + 0.022 * h2) * scale
        lines.append(f'      <geom type="cylinder" pos="{px:.3f} {py:.3f} {pz + sh * 0.3:.3f}" size="{sr:.3f} {sh:.3f}" rgba="{col}" contype="0" conaffinity="0"/>')
      elif kind == 2:  # capsule bottle / tool
        sr, sh = (0.014 + 0.008 * h1) * scale, (0.024 + 0.018 * h2) * scale
        lines.append(f'      <geom type="capsule" pos="{px:.3f} {py:.3f} {pz + sh * 0.3:.3f}" {rot} size="{sr:.3f} {sh:.3f}" rgba="{col}" contype="0" conaffinity="0"/>')
      elif kind == 3:  # sphere fruit / canister
        sr = (0.022 + 0.012 * h1) * scale
        lines.append(f'      <geom type="sphere" pos="{px:.3f} {py:.3f} {pz:.3f}" size="{sr:.3f}" rgba="{col}" contype="0" conaffinity="0"/>')
      else:  # ellipsoid bowl / jar
        sx, sy, sz = (0.028 + 0.012 * h1) * scale, (0.024 + 0.010 * h2) * scale, (0.016 + 0.010 * h3) * scale
        lines.append(f'      <geom type="ellipsoid" pos="{px:.3f} {py:.3f} {pz:.3f}" {rot} size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="{col}" contype="0" conaffinity="0"/>')
      idx_global += 1
      prim_count += 1

    lines.append('    </body>')

  lines.append('  </worldbody>')
  lines.append('  <actuator>')
  for prefix in ("left_arm", "right_arm"):
    for j in ("j1", "j2", "j3", "j4"):
      lines.append(f'    <position name="{prefix}_{j}_act" joint="{prefix}_{j}" kp="80" kv="5" ctrlrange="-1.5 1.5"/>')
  for zname, _, _, _, _, _ in zones:
    lines.append(f'    <motor name="{zname}_act" joint="{zname}_j" gear="5" ctrlrange="-1 1"/>')
  lines.append('  </actuator>')
  lines.append('</mujoco>')
  return "\n".join(lines)
