/* ==========================================================================
   TPU-RTX Bilingual Content Dictionary (English / 中文)
   Pure Single-Scene Architecture:
   - 23 Pure Standalone Single-Scene Plates + 2 Full-Width QA-Rejected Cases
   - 3 Interactive Split Slider Labs (19 Aligned Left/Right Presets)
   - 2 Interactive Full-Width Single-Scene Stages (3DGS 4K & Menagerie PBR)
   - GPU vs TPU Tradeoffs, 4 Post-Mortems, Ablation Table, 20-Row JSON Ledger
   Author: Feiyang Liu
   ========================================================================== */

window.TPU_RTX_DATA = {
  plates: [
    // --- Moana & Film / Caustics Singles ---
    {
      id: "moana_sunset",
      cat: "film",
      web: "./assets/singles/moana_sunset.webp",
      png: "./assets/singles/moana_sunset.webp",
      badges: {
        en: [["hot", "22.30B Instanced Tris"], ["", "Native 4K"], ["ok", "4.65 GiB HBM · 5.60 s"]],
        zh: [["hot", "223 亿实例化三角面"], ["", "原生 4K"], ["ok", "4.65 GiB 显存 · 5.60 s"]]
      },
      title: {
        en: "Disney Moana OpenUSD v2.1: Motunui Golden Sunset Lagoon (4K)",
        zh: "迪士尼 Moana OpenUSD v2.1：Motunui 黄金日落海湾（原生 4K）"
      },
      desc: {
        en: "45.5M unique + 14.0M instanced BLAS triangles packed into 4.653 GiB HBM on Google Cloud TPU v5p: Rayleigh atmospheric sunset glare and specular ocean wave glints traced in 5.60 s @ 3840x2160 with zero host callbacks.",
        zh: "4,550 万唯一三角面 + 1,400 万实例化 BLAS（总等效 223 亿面），仅占 4.653 GiB HBM：Rayleigh 黄金日落大气散射与海面波光反射，3840x2160 单帧 5.60 s、全程零主机回调。"
      }
    },
    {
      id: "moana_reef",
      cat: "film",
      web: "./assets/singles/moana_reef.webp",
      png: "./assets/singles/moana_reef.webp",
      badges: {
        en: [["hot", "Underwater Reef 4K"], ["", "16-Step HG God Rays"], ["ok", "8.72 s @ 4K"]],
        zh: [["hot", "水下珊瑚礁 4K"], ["", "16 步 HG 圣光束"], ["ok", "4K 8.72 s"]]
      },
      title: {
        en: "Disney Moana OpenUSD v2.1: Submerged Coral Reef & Snell's Window (4K)",
        zh: "迪士尼 Moana OpenUSD v2.1：水下珊瑚礁、Snell 全反射天窗与焦散（4K）"
      },
      desc: {
        en: "Beer-Lambert cyan-emerald absorption, overhead Snell total-internal-reflection window, seafloor wave caustics, and 16-step Henyey-Greenstein volumetric shafts rendered natively inside TPU HBM.",
        zh: "Beer-Lambert 蓝绿水体吸收、水面 Snell 全反射天窗、海底波纹焦散与 16 步 Henyey-Greenstein 体积丁达尔光束，完全在 TPU HBM 内联算完成。"
      }
    },
    {
      id: "moana_mjx_beach",
      cat: "film",
      web: "./assets/singles/moana_mjx_beach.webp",
      png: "./assets/singles/moana_mjx_beach.webp",
      badges: {
        en: [["hot", "Physical AI x Film"], ["", "22.30B + 45 MJX Geoms"], ["ok", "5.40 s @ 4K"]],
        zh: [["hot", "具身智能 x 电影世界"], ["", "223 亿面 + 45 刚体"], ["ok", "4K 5.40 s"]]
      },
      title: {
        en: "Live MJX Humanoid & Quadruped Running on Disney Moana Beach (4K)",
        zh: "MJX 人形与四足机器人直接奔跑在迪士尼 Moana 沙滩（原生 4K）"
      },
      desc: {
        en: "Single unified TLAS traces both Disney's 22.30B-tri Moana OpenUSD island and 45 live MuJoCo MJX robot geoms with mutual BVH sand shadows inside one XLA graph (host_callbacks = 0).",
        zh: "单个统一 TLAS 在同一张 XLA 计算图内同时追踪 223 亿面 Moana OpenUSD 世界与 45 个实时驱动的 MuJoCo MJX 关节刚体，并在沙滩上投下互影。"
      }
    },
    {
      id: "moana_8k_bay",
      cat: "film",
      web: "./assets/singles/moana_8k_bay.webp",
      png: "./assets/singles/moana_8k_bay.webp",
      badges: {
        en: [["hot", "Native 8K UHD"], ["", "7680x4320 · 33.18M px"], ["ok", "17.24 s / 4 Chips"]],
        zh: [["hot", "原生 8K UHD"], ["", "7680x4320 · 3,318 万像素"], ["ok", "4 芯 17.24 s"]]
      },
      title: {
        en: "Disney Moana OpenUSD @ Native 8K Full-Bay Aerial Overview",
        zh: "迪士尼 Moana OpenUSD 原生 8K（7680x4320）海湾俯瞰全景"
      },
      desc: {
        en: "Full 33.18M-pixel 8K raytrace directly over OpenUSD PointInstancers across 4 TPU v5p chips in 17.24 s (3.85 MRays/s, gather_ops = 0).",
        zh: "4 颗 TPU v5p 芯片协同直出完整 7680x4320 像素 8K 全景海湾，直接遍历 OpenUSD PointInstancer（17.24 s，gather_ops = 0）。"
      }
    },
    // --- Standalone Blender Studio TPU-RTX Plates ---
    {
      id: "bmw27_tpu",
      cat: "film",
      web: "./assets/singles/bmw27_tpu.webp",
      png: "./assets/singles/bmw27_tpu.webp",
      badges: {
        en: [["hot", "Blender BMW27"], ["", "37.65 dB Orbit"], ["ok", "0.9813 SSIM"]],
        zh: [["hot", "Blender BMW27"], ["", "37.65 dB 环绕"], ["ok", "SSIM 0.9813"]]
      },
      title: {
        en: "Blender Open Data BMW27 Clearcoat Car (TPU-RTX AgX Compilation)",
        zh: "Blender 官方 BMW27 双层清漆跑车（TPU-RTX 直编 + AgX）"
      },
      desc: {
        en: "Compiled directly from official .blend Cycles shader graph to Wide-BVH-64: 37.65 dB orbit PSNR, 0.9813 SSIM, and -0.012% luminance bias vs Cycles EXR.",
        zh: "直接由官方 .blend Cycles 节点图编译至 Wide-BVH-64：环绕视角 37.65 dB、SSIM 0.9813、亮度偏差仅 -0.012%。"
      }
    },
    {
      id: "barcelona_tpu",
      cat: "film",
      web: "./assets/singles/barcelona_tpu.webp",
      png: "./assets/singles/barcelona_tpu.webp",
      badges: {
        en: [["hot", "Barcelona 54.91M Tris"], ["", "15.94 mm Z-MAE"], ["ok", "1.11 img/s @ 4K"]],
        zh: [["hot", "巴塞罗那馆 5,491 万面"], ["", "15.94 mm 深度误差"], ["ok", "4K 1.11 img/s"]]
      },
      title: {
        en: "Blender Barcelona Pavilion (54.91M Unique Triangles, Cycles Parity)",
        zh: "Blender 巴塞罗那展馆（5,491 万唯一三角面，Cycles 对齐）"
      },
      desc: {
        en: "54.91M architectural & foliage triangles traversed with 15.94 mm Z-depth MAE and 1.11 img/s Native 4K throughput on 4 TPU v5p chips.",
        zh: "5,491 万建筑与高密树叶三角面在 4 芯 TPU v5p 上实现 15.94 mm 深度误差与 4K 1.11 img/s。"
      }
    },
    {
      id: "monster_sss_tpu",
      cat: "film",
      web: "./assets/singles/monster_sss_tpu.webp",
      png: "./assets/singles/monster_sss_tpu.webp",
      badges: {
        en: [["hot", "Monster SSS 1024²"], ["", "36.98 dB Orbit"], ["ok", "0.121 mm Z-MAE"]],
        zh: [["hot", "床下怪兽 SSS"], ["", "36.98 dB 环绕"], ["ok", "0.121 mm 深度差"]]
      },
      title: {
        en: "Blender Studio Monster Under the Bed SSS (1.60M Tris, 10.07 img/s)",
        zh: "Blender Studio 床下怪兽 SSS（160 万面，4 芯 1024² 达 10.07 img/s）"
      },
      desc: {
        en: "Subsurface-scattered character asset hitting 36.98 dB orbit (35.34 dB hero) and 0.121 mm geometric Z-MAE at 10.07 img/s (1024²).",
        zh: "次表面散射角色资产达成 36.98 dB 环绕（特写 35.34 dB）、0.121 mm 几何深度误差与 1024² 10.07 img/s。"
      }
    },
    {
      id: "fishycat_tpu",
      cat: "film",
      web: "./assets/singles/fishycat_tpu.webp",
      png: "./assets/singles/fishycat_tpu.webp",
      badges: {
        en: [["hot", "Fishy Cat Fur"], ["", "7,255 Fur Instances"], ["ok", "28.80 dB / 0.9592 SSIM"]],
        zh: [["hot", "Fishy Cat 毛发池塘"], ["", "7,255 毛发实例"], ["ok", "28.80 dB / 0.9592 SSIM"]]
      },
      title: {
        en: "Blender Studio Fishy Cat (907k Tris, 7,255 Instanced Fur Strands)",
        zh: "Blender Studio Fishy Cat（90.7 万面，7,255 根实例化毛发）"
      },
      desc: {
        en: "Multi-material fur, translucent water, and grass courtyard reaching 28.80 dB PSNR and 0.9592 SSIM against Blender Cycles.",
        zh: "多材质绒毛、折射水面与庭院草丛达成 28.80 dB PSNR 与 0.9592 SSIM。"
      }
    },
    {
      id: "barbershop_tpu",
      cat: "film",
      web: "./assets/singles/barbershop_tpu.webp",
      png: "./assets/singles/barbershop_tpu.webp",
      badges: {
        en: [["hot", "8.55M Tris · 141 Tex"], ["", "31.00 dB Orbit"], ["ok", "13.8 mm MedAE"]],
        zh: [["hot", "理发店 855 万面"], ["", "141 张纹理 · 31.00 dB"], ["ok", "13.8 mm 中位误差"]]
      },
      title: {
        en: "Blender Barbershop Interior (8.55M Tris, 249 Materials, 141 Textures)",
        zh: "Blender 理发店复杂室内（855 万面、249 材质、141 张打包贴图）"
      },
      desc: {
        en: "Heavy architectural interior with transparent-glass blocker handling and 141 packed UV maps achieving 31.00 dB orbit (SSIM 0.9285, 4K 1.16 img/s).",
        zh: "包含玻璃窗透射与 141 张打包 UV 贴图的经典硬核室内基准：31.00 dB 环绕（SSIM 0.9285，4K 1.16 img/s）。"
      }
    },
    {
      id: "lonemonk_tpu",
      cat: "film",
      web: "./assets/singles/lonemonk_tpu.webp",
      png: "./assets/singles/lonemonk_tpu.webp",
      badges: {
        en: [["hot", "Lone Monk 20.14M"], ["", "28.83 dB Orbit"], ["ok", "3.58 img/s @ 1080p"]],
        zh: [["hot", "独行僧侣 2,014 万面"], ["", "28.83 dB 环绕"], ["ok", "1080p 3.58 img/s"]]
      },
      title: {
        en: "Blender Open Data Lone Monk Courtyard, Lanterns & Dense Foliage",
        zh: "Blender 官方独行僧侣庭院（密集树冠与灯笼暖光）"
      },
      desc: {
        en: "20.14M instanced canopy triangles with warm lantern bounce lighting compiled to Wide-BVH-64 at 28.83 dB orbit and 18.91 mm Z-MAE.",
        zh: "2,014 万等效实例化植被面与灯笼多材质漫反射暖光：28.83 dB 环绕、18.91 mm 深度误差。"
      }
    },
    // --- Physical Optics Singles ---
    {
      id: "glass_caustics",
      cat: "optics",
      web: "./assets/singles/glass_caustics.webp",
      png: "./assets/singles/glass_caustics.webp",
      badges: {
        en: [["hot", "5-Bounce Caustics"], ["", "Beer-Lambert Glass"], ["ok", "192 spp · 0 Host CB"]],
        zh: [["hot", "5 次反弹玻璃焦散"], ["", "Beer-Lambert 介质吸收"], ["ok", "192 spp · 零回调"]]
      },
      title: {
        en: "Dielectric Crown Glass Lens Caustics & Beer-Lambert Concentrator",
        zh: "电介质光学玻璃球焦散与 Beer-Lambert 透射聚光（192 spp）"
      },
      desc: {
        en: "Pure dielectric Fresnel refraction concentrating directional skylight into sharp floor caustic arcs inside one branchless XLA path tracer.",
        zh: "5 次纯物理电介质 Fresnel 折射将高光汇聚成地面明锐焦散弧线，整条蒙特卡洛路径循环零主机回调。"
      }
    },
    {
      id: "prism_dispersion",
      cat: "optics",
      web: "./assets/singles/prism_dispersion.webp",
      png: "./assets/singles/prism_dispersion.webp",
      badges: {
        en: [["hot", "6-Band Cauchy Prism"], ["", "Wavelength IOR"], ["ok", "128 spp Pure PT"]],
        zh: [["hot", "6 波段柯西棱镜色散"], ["", "分波长折射率"], ["ok", "128 spp 纯路径追踪"]]
      },
      title: {
        en: "Spectral Cauchy Prism Dispersion (6 Wavelength Bins)",
        zh: "6 波段柯西光谱三棱镜色散（物理分波长折射）"
      },
      desc: {
        en: "Wavelength-varying index of refraction splits white collimated light into clean RGB spectral caustic ribbons on the floor.",
        zh: "按波长查表的玻璃折射率将白光束精准色散劈开为红橙黄绿青蓝光谱焦散带。"
      }
    },
    {
      id: "jade_sss",
      cat: "optics",
      web: "./assets/singles/jade_sss.webp",
      png: "./assets/singles/jade_sss.webp",
      badges: {
        en: [["hot", "16-Step Random-Walk"], ["", "Volumetric Jade SSS"], ["ok", "128 spp · 0 Gather"]],
        zh: [["hot", "16 步随机游走 SSS"], ["", "半透明玉石次表面散射"], ["ok", "128 spp · 零 Gather"]]
      },
      title: {
        en: "16-Step Random-Walk Subsurface Scattering in Translucent Jade",
        zh: "16 步体内随机游走玉石次表面散射（Translucency）"
      },
      desc: {
        en: "Volumetric mean-free-path stepping inside carved meshes produces warm inner green-emerald glow under grazing backlight.",
        zh: "真实体内平均自由程随机游走步进，使薄壁与背光曲面透出柔和温润的翡翠内透光感。"
      }
    },
    {
      id: "cathedral_godrays",
      cat: "optics",
      web: "./assets/singles/cathedral_godrays.webp",
      png: "./assets/singles/cathedral_godrays.webp",
      badges: {
        en: [["hot", "Volumetric God Rays"], ["", "BVH-Shadowed Fog"], ["ok", "192 spp XLA Loop"]],
        zh: [["hot", "教堂体积丁达尔光"], ["", "BVH 阴影体积散射"], ["ok", "192 spp 单图循环"]]
      },
      title: {
        en: "BVH-Shadowed Volumetric Cathedral Crepuscular Rays",
        zh: "BVH 遮挡阴影下的教堂体积圣光束（Crepuscular Rays）"
      },
      desc: {
        en: "Equi-angular volumetric ray marching coupled with stained-glass BVH shadow queries inside TPU HBM.",
        zh: "等角度体积采样与宽 BVH 阴影查询紧密耦合，全流程在单张 XLA 计算图内直出体积光柱。"
      }
    },
    // --- Real-World 3DGS Singles (Used in Ch 04 Stage) ---
    {
      id: "garden_4k",
      cat: "gs3d",
      web: "./assets/singles/garden_4k.webp",
      png: "./assets/singles/garden_4k.webp",
      badges: {
        en: [["hot", "5.83M Real SH3"], ["", "4K 2.61 FPS · 8K 0.88"], ["ok", "gather_ops = 0"]],
        zh: [["hot", "583 万真实 SH3 高斯"], ["", "4K 2.61 / 8K 0.88 FPS"], ["ok", "gather_ops = 0"]]
      },
      title: {
        en: "Mip-NeRF 360 garden (5,834,784 SH3 Gaussians @ Native 4K / 8K)",
        zh: "Mip-NeRF 360 garden（583 万全 SH3 航拍高斯，原生 4K / 8K）"
      },
      desc: {
        en: "Genuine .ply photogrammetry scan rasterized without any HBM gather: 2.61 FPS @ Native 4K (3840x2160) and 29.19 MPix/s @ Native 8K (7680x4320) across 4 TPU v5p chips.",
        zh: "真实 .ply 航拍重建高斯场，全程零动态 HBM Gather：4 芯原生 4K 跑 2.61 FPS，原生 8K 跑 29.19 MPix/s。"
      }
    },
    {
      id: "bicycle_4k",
      cat: "gs3d",
      web: "./assets/singles/bicycle_4k.webp",
      png: "./assets/singles/bicycle_4k.webp",
      badges: {
        en: [["hot", "6.13M Real SH3"], ["", "Native 4K 2.59 FPS"], ["ok", "0 Gather Ops"]],
        zh: [["hot", "613 万真实 SH3 高斯"], ["", "原生 4K 2.59 FPS"], ["ok", "零 Gather"]]
      },
      title: {
        en: "Mip-NeRF 360 bicycle (6,131,954 SH3 Gaussians @ Native 4K)",
        zh: "Mip-NeRF 360 bicycle（613 万 SH3 高斯户外公园 @ 原生 4K）"
      },
      desc: {
        en: "6.13M anisotropic splats over park grass, spokes, and bench rendered at full 3840x2160 in 386 ms on 4 chips.",
        zh: "613 万细草坪与自行车辐条高斯体在 4 芯下直出完整 3840x2160，仅需 386 ms（2.59 FPS）。"
      }
    },
    {
      id: "kitchen_4k",
      cat: "gs3d",
      web: "./assets/singles/kitchen_4k.webp",
      png: "./assets/singles/kitchen_4k.webp",
      badges: {
        en: [["hot", "1.80M Real SH3"], ["", "Native 4K 3.98 FPS"], ["ok", "Host of 4DGS Push"]],
        zh: [["hot", "180 万真实 SH3 高斯"], ["", "原生 4K 3.98 FPS"], ["ok", "4DGS 物理交互底图"]]
      },
      title: {
        en: "Mip-NeRF 360 kitchen + LEGO Dozer (Native 4K @ 3.98 FPS)",
        zh: "Mip-NeRF 360 kitchen 厨房乐高实景（原生 4K 3.98 FPS）"
      },
      desc: {
        en: "1.80M SH3 splats rendered at Native 4K in 251 ms (3.98 FPS) — and serves as the live background for Franka Panda 4DGS collision pushing.",
        zh: "180 万 SH3 高斯在 4K 跑出 3.98 FPS（251 ms），同时也是下方 4DGS 机械臂撞击乐高推土机实验的真实场景。"
      }
    },
    {
      id: "bonsai_4k",
      cat: "gs3d",
      web: "./assets/singles/bonsai_4k.webp",
      png: "./assets/singles/bonsai_4k.webp",
      badges: {
        en: [["hot", "1.24M Real SH3"], ["", "Native 4K 4.20 FPS"], ["ok", "238 ms / Frame"]],
        zh: [["hot", "124 万真实 SH3 高斯"], ["", "原生 4K 4.20 FPS"], ["ok", "238 ms / 帧"]]
      },
      title: {
        en: "Mip-NeRF 360 bonsai Indoor Scan (Native 4K @ 4.20 FPS)",
        zh: "Mip-NeRF 360 bonsai 室内盆栽扫描（原生 4K 4.20 FPS）"
      },
      desc: {
        en: "Fine bonsai needle leaves and specular table reflections at full 3840x2160 resolution in 238 ms on 4 TPU v5p chips.",
        zh: "细腻松针与桌面反射在 4 芯 TPU v5p 上全分辨率 3840x2160 仅 238 ms（4.20 FPS）。"
      }
    },
    {
      id: "bistro_humanoid_4k",
      cat: "gs3d",
      web: "./assets/singles/bistro_humanoid_4k.webp",
      png: "./assets/singles/bistro_humanoid_4k.webp",
      badges: {
        en: [["hot", "10M Bistro + MJX"], ["", "Native 4K 2.28 FPS"], ["ok", "Exact Z-Composite"]],
        zh: [["hot", "1,000 万 Bistro + MJX"], ["", "原生 4K 2.28 FPS"], ["ok", "精准 Z-Buffer 混合"]]
      },
      title: {
        en: "10M-Gaussian Amazon Bistro x Live MJX Humanoid & Mobile Panda (4K)",
        zh: "1,000 万高斯 Amazon Bistro 街区 x MJX 人形与移动熊猫机器人（4K）"
      },
      desc: {
        en: "10M-splat European street environment composited pixel-for-pixel via Z-depth with live articulated MJX humanoid & mobile manipulator at 439.3 ms @ 4K.",
        zh: "1,000 万街区 3DGS 与实时 MJX 人形/熊猫双机器人 4K 逐像素深度融合，4 芯单帧 439.3 ms（2.28 FPS）。"
      }
    },
    // --- MuJoCo Menagerie PBR Robot Singles (Used in Ch 05 Stage) ---
    {
      id: "menagerie_g1",
      cat: "robotics",
      web: "./assets/singles/menagerie_g1.webp",
      png: "./assets/singles/menagerie_g1.webp",
      badges: {
        en: [["hot", "Unitree G1 (393k tris)"], ["", "2,546 FPS @ 64² (4c)"], ["ok", "1024² PBR Hero"]],
        zh: [["hot", "宇树 G1 (39.3 万面)"], ["", "4 芯 64² 2,546 FPS"], ["ok", "1024² PBR 实测"]]
      },
      title: {
        en: "Unitree G1 Humanoid — Unmodified Menagerie MJCF in Wide-BVH-64",
        zh: "宇树 Unitree G1 人形机器人 — 原生 Menagerie MJCF 直编上 TPU"
      },
      desc: {
        en: "393k mesh triangles from official DeepMind mujoco_menagerie compiled directly into MultiMeshBVHBank: 2,546 FPS (64², 4c) + 1024² GGX+8-AO+8-SoftShadow hero.",
        zh: "官方 mujoco_menagerie 39.3 万面高精度网格零修改编译入片内：4 芯 64² 吞吐 2,546 FPS，1024² GGX+8-AO 全特效渲染。"
      }
    },
    {
      id: "menagerie_go2",
      cat: "robotics",
      web: "./assets/singles/menagerie_go2.webp",
      png: "./assets/singles/menagerie_go2.webp",
      badges: {
        en: [["hot", "Unitree Go2 (398k tris)"], ["", "1,863 FPS @ 64² (4c)"], ["ok", "Zero Gather"]],
        zh: [["hot", "宇树 Go2 (39.8 万面)"], ["", "4 芯 64² 1,863 FPS"], ["ok", "零 HBM Gather"]]
      },
      title: {
        en: "Unitree Go2 Quadruped — Official Menagerie MJCF + GGX Soft Shadows",
        zh: "宇树 Unitree Go2 四足机器狗 — 官方 Menagerie MJCF + GGX 软阴影"
      },
      desc: {
        en: "398k curved chassis & actuator triangles running live MJX forward kinematics inside TPU HBM at 1,863 FPS (64², 4c) with zero CPU sync.",
        zh: "39.8 万细腻曲面电机外壳三角面与 MJX 运动学链同驻 HBM，4 芯 64² 跑 1,863 FPS、零主机回传。"
      }
    },
    {
      id: "menagerie_panda",
      cat: "robotics",
      web: "./assets/singles/menagerie_panda.webp",
      png: "./assets/singles/menagerie_panda.webp",
      badges: {
        en: [["hot", "Franka Panda Arm"], ["", "3,131 FPS @ 64² (4c)"], ["ok", "jax.grad Compatible"]],
        zh: [["hot", "Franka Panda 机械臂"], ["", "4 芯 64² 3,131 FPS"], ["ok", "支持端到端 jax.grad"]]
      },
      title: {
        en: "Franka Emika Panda 7-DOF Arm — Menagerie PBR + Soft Shadows",
        zh: "Franka Emika Panda 7 自由度机械臂 — 官方 Menagerie PBR 渲染"
      },
      desc: {
        en: "Standard manipulation testbed compiled directly from Menagerie MJCF: 3,131 FPS zero-copy throughput and end-to-end differentiable visual servoing.",
        zh: "具身操控经典标准臂：零拷贝 3,131 FPS，同时支持第 05 章的纯视觉像素到力矩可微伺服。"
      }
    },
    {
      id: "menagerie_aerohand",
      cat: "robotics",
      web: "./assets/singles/menagerie_aerohand.webp",
      png: "./assets/singles/menagerie_aerohand.webp",
      badges: {
        en: [["hot", "Tetheria Aero Hand"], ["", "4,284 FPS @ 64² (4c)"], ["ok", "Articulated Fingers"]],
        zh: [["hot", "Tetheria Aero Hand 灵巧手"], ["", "4 芯 64² 4,284 FPS"], ["ok", "高自由度指节"]]
      },
      title: {
        en: "Tetheria Aero Hand Dexterous Hand — 4,284 FPS Multi-Link Raytracing",
        zh: "Tetheria Aero Hand 灵巧手 — 4,284 FPS 多关节高频光追"
      },
      desc: {
        en: "Multi-finger tendon manipulator traced natively inside Wide-BVH-64 at 4,284 FPS (64², 4c) for high-throughput dexterous policy & GelSight research.",
        zh: "多指灵巧手在 Wide-BVH-64 内跑出 4,284 FPS（4 芯），直接服务高速触觉与灵巧手策略训练。"
      }
    },
    // --- 2 QA-Rejected Post-Mortem Singles ---
    {
      id: "rejected_openusd_clay",
      cat: "rejected",
      web: "./assets/postmortem/rejected_openusd_clay.webp",
      png: "./assets/postmortem/rejected_openusd_clay.webp",
      badges: {
        en: [["fail", "QA REJECTED"], ["", "Early OpenUSD Bug"], ["", "Flat Clay Shading"]],
        zh: [["fail", "QA 已拦截淘汰"], ["", "早期 OpenUSD 缺陷"], ["", "平淡白模着色"]]
      },
      title: {
        en: "QA Rejected Case A: Early Untextured OpenUSD Stage (v3_w2_openusd_4k)",
        zh: "QA 拦截淘汰案例 A：早期未完整绑定 Ptex 的“白模泥塑感”（v3_w2_openusd_4k）"
      },
      desc: {
        en: "Before full Disney Ptex & BSDF material binding, distant tropical foliage and lagoon waves collapsed into low-contrast flat clay. Rejected by Visual QA and replaced by the Sunset, Coral Reef, and Moana x MJX passes above.",
        zh: "在完整接入迪士尼 Ptex 纹理与水面 Fresnel/PBR 材质编译之前，远景植被与海浪曾坍缩成低对比度白模。被视觉 QA 直接驳回，后被第 03 章三条黄金 4K 成片取代。"
      }
    },
    {
      id: "rejected_garden_seams",
      cat: "rejected",
      web: "./assets/postmortem/rejected_garden_seams.webp",
      png: "./assets/postmortem/rejected_garden_seams.webp",
      badges: {
        en: [["fail", "QA REJECTED"], ["", "16x16 Tile Seams"], ["", "31.31 dB Numerics OK"]],
        zh: [["fail", "QA 已拦截淘汰"], ["", "16x16 分块拼缝"], ["", "仅保留数值日志"]]
      },
      title: {
        en: "QA Rejected Case B: Early 32-View 3DGS Local Frustum Clipping (garden)",
        zh: "QA 拦截淘汰案例 B：早期 32 视角 3DGS 局部视锥截断拼缝（garden）"
      },
      desc: {
        en: "Early local tile bounds clipped Gaussian tails at 16x16 boundaries, yielding visible grid blocks and masked borders. Excluded from curated showcases while keeping the verified 31.31 dB gradient telemetry.",
        zh: "早期局部 tile 累积边界截断了跨 16x16 块的高斯尾迹，产生明显方块拼缝与黑边。成片已淘汰并公开展示反面样本，对应 31.31 dB 数值训练日志已单独验证修正。"
      }
    }
  ],

  spreadCh3: ["moana_sunset", "moana_reef", "moana_mjx_beach", "moana_8k_bay", "glass_caustics"],

  stage3dgs: [
    {
      plateId: "garden_4k",
      tabIdx: { en: "SCENE 01 · OUTDOOR", zh: "场景 01 · 户外花园" },
      tabTitle: { en: "garden (5.83M SH3 · 2.61 FPS)", zh: "garden 航拍花园 (583 万高斯)" },
      hudL: { en: "<strong>Mip-NeRF 360 garden</strong> · 5,834,784 SH3 Splats · gather_ops = 0", zh: "<strong>Mip-NeRF 360 garden</strong> · 583 万全 SH3 高斯 · 零 Gather" },
      hudR: { en: "Native <strong>4K @ 2.61 FPS</strong> · <strong>8K @ 29.19 MPix/s</strong> · Click to Zoom", zh: "原生 <strong>4K 2.61 FPS</strong> · <strong>8K 29.19 MPix/s</strong> · 点击放大" }
    },
    {
      plateId: "bicycle_4k",
      tabIdx: { en: "SCENE 02 · FOLIAGE", zh: "场景 02 · 细草坪与辐条" },
      tabTitle: { en: "bicycle (6.13M SH3 · 2.59 FPS)", zh: "bicycle 自行车 (613 万高斯)" },
      hudL: { en: "<strong>Mip-NeRF 360 bicycle</strong> · 6,131,954 SH3 Splats · 4 TPU v5p Chips", zh: "<strong>Mip-NeRF 360 bicycle</strong> · 613 万全 SH3 高斯 · 4 芯 TPU v5p" },
      hudR: { en: "Native <strong>4K (3840x2160) @ 2.59 FPS</strong> (386 ms) · Click to Zoom", zh: "原生 <strong>4K (3840x2160) 2.59 FPS</strong> (386 ms) · 点击放大" }
    },
    {
      plateId: "kitchen_4k",
      tabIdx: { en: "SCENE 03 · INDOOR LEGO", zh: "场景 03 · 室内厨房" },
      tabTitle: { en: "kitchen (1.80M SH3 · 3.98 FPS)", zh: "kitchen 厨房乐高 (180 万高斯)" },
      hudL: { en: "<strong>Mip-NeRF 360 kitchen</strong> · 1,802,911 SH3 Splats · 4DGS Contact Host", zh: "<strong>Mip-NeRF 360 kitchen</strong> · 180 万 SH3 高斯 · 4DGS 物理推挤底图" },
      hudR: { en: "Native <strong>4K @ 3.98 FPS</strong> (251 ms) · Click to Zoom", zh: "原生 <strong>4K 3.98 FPS</strong> (251 ms) · 点击放大" }
    },
    {
      plateId: "bonsai_4k",
      tabIdx: { en: "SCENE 04 · BONSAI", zh: "场景 04 · 室内盆栽" },
      tabTitle: { en: "bonsai (1.24M SH3 · 4.20 FPS)", zh: "bonsai 盆栽 (124 万高斯)" },
      hudL: { en: "<strong>Mip-NeRF 360 bonsai</strong> · 1,243,210 SH3 Splats · Finer Pine Foliage", zh: "<strong>Mip-NeRF 360 bonsai</strong> · 124 万 SH3 高斯 · 细密松针扫描" },
      hudR: { en: "Native <strong>4K @ 4.20 FPS</strong> (238 ms) · Click to Zoom", zh: "原生 <strong>4K 4.20 FPS</strong> (238 ms) · 点击放大" }
    },
    {
      plateId: "bistro_humanoid_4k",
      tabIdx: { en: "SCENE 05 · 10M STREET", zh: "场景 05 · 千万街区+人形" },
      tabTitle: { en: "10M Bistro × MJX (2.28 FPS 4K)", zh: "1,000 万 Bistro × MJX (2.28 FPS)" },
      hudL: { en: "<strong>10M Bistro + MJX Humanoid &amp; Mobile Panda</strong> · Exact Z-Composite", zh: "<strong>1,000 万 Bistro + MJX 人形/熊猫双机器人</strong> · 逐像素深度融合" },
      hudR: { en: "Native <strong>4K @ 2.28 FPS</strong> (439.3 ms, <strong>5.24x</strong> Speedup) · Click to Zoom", zh: "原生 <strong>4K 2.28 FPS</strong> (439.3 ms, 提速 <strong>5.24 倍</strong>) · 点击放大" }
    }
  ],

  stageMenagerie: [
    {
      plateId: "menagerie_g1",
      tabIdx: { en: "ROBOT 01 · HUMANOID", zh: "机器人 01 · 双足人形" },
      tabTitle: { en: "Unitree G1 (393k · 2,546 FPS)", zh: "宇树 G1 人形 (2,546 FPS)" },
      hudL: { en: "<strong>Unitree G1 Humanoid</strong> · 393k Mesh Tris · Wide-BVH-64 + 1024² GGX", zh: "<strong>宇树 Unitree G1 人形机器人</strong> · 39.3 万面 · Wide-BVH-64" },
      hudR: { en: "<strong>2,546 FPS</strong> @ 64² (4 Chips, host_callbacks=0) · Click to Zoom", zh: "4 芯 64² 实测 <strong>2,546 FPS</strong>（零回调）· 点击放大" }
    },
    {
      plateId: "menagerie_go2",
      tabIdx: { en: "ROBOT 02 · QUADRUPED", zh: "机器人 02 · 四足机器狗" },
      tabTitle: { en: "Unitree Go2 (398k · 1,863 FPS)", zh: "宇树 Go2 四足 (1,863 FPS)" },
      hudL: { en: "<strong>Unitree Go2 Quadruped</strong> · 398k Mesh Tris · Curved Chassis + Soft Shadow", zh: "<strong>宇树 Unitree Go2 四足机器狗</strong> · 39.8 万面 · 软阴影 PBR" },
      hudR: { en: "<strong>1,863 FPS</strong> @ 64² (4 Chips, gather_ops=0) · Click to Zoom", zh: "4 芯 64² 实测 <strong>1,863 FPS</strong>（零 Gather）· 点击放大" }
    },
    {
      plateId: "menagerie_panda",
      tabIdx: { en: "ROBOT 03 · 7-DOF ARM", zh: "机器人 03 · 7 轴机械臂" },
      tabTitle: { en: "Franka Panda (3,131 FPS)", zh: "Franka Panda 机械臂 (3,131 FPS)" },
      hudL: { en: "<strong>Franka Emika Panda</strong> · Menagerie MJCF · Diff Physics Servoing Target", zh: "<strong>Franka Emika Panda</strong> · 官方 Menagerie · 可微视觉伺服平台" },
      hudR: { en: "<strong>3,131 FPS</strong> @ 64² (4 Chips) · <code>jax.grad</code> 36.08 dB · Click to Zoom", zh: "4 芯 64² <strong>3,131 FPS</strong> · <code>jax.grad</code> 伺服 36.08 dB · 点击放大" }
    },
    {
      plateId: "menagerie_aerohand",
      tabIdx: { en: "ROBOT 04 · DEXTEROUS", zh: "机器人 04 · 多指灵巧手" },
      tabTitle: { en: "Aero Hand (4,284 FPS)", zh: "Aero Hand 灵巧手 (4,284 FPS)" },
      hudL: { en: "<strong>Tetheria Aero Hand</strong> · Articulated Fingers · Tactile &amp; Manipulation Ready", zh: "<strong>Tetheria Aero Hand 灵巧手</strong> · 多指关节 · 视触觉策略高速底座" },
      hudR: { en: "<strong>4,284 FPS</strong> @ 64² (4 Chips) · Click to Zoom", zh: "4 芯 64² 实测 <strong>4,284 FPS</strong> · 点击放大" }
    }
  ],

  heroStage: [
    {
      plateId: "moana_sunset",
      hudL: {
        en: "<strong>Disney Moana OpenUSD v2.1</strong> · 22.30B Instanced Tris in 4.65 GiB HBM · host_callbacks=0",
        zh: "<strong>迪士尼 Moana OpenUSD v2.1</strong> · 223 亿面仅占 4.65 GiB HBM · 零回调"
      },
      hudR: {
        en: "Native <strong>4K (3840x2160)</strong> in <strong>5.60 s</strong> · Click frame for Lightbox + 1:1 Loupe",
        zh: "原生 <strong>4K (3840x2160)</strong> 单帧 <strong>5.60 s</strong> · 点击画面打开 1:1 放大镜"
      }
    },
    {
      plateId: "moana_mjx_beach",
      hudL: {
        en: "<strong>Physical AI x Film OpenUSD</strong> · Live MJX Humanoid &amp; Quadruped on Motunui Beach",
        zh: "<strong>具身智能 x 电影 OpenUSD</strong> · MJX 人形与四足机器人实景互影奔跑"
      },
      hudR: {
        en: "Single unified TLAS · Native <strong>4K in 5.40 s</strong> · gather_ops=0",
        zh: "单个统一 TLAS · 原生 <strong>4K 5.40 s</strong> · 零 HBM Gather"
      }
    },
    {
      plateId: "garden_4k",
      hudL: {
        en: "<strong>Real-World Mip-NeRF 360 garden</strong> · 5.83M SH3 Gaussians · Native EWA in VMEM",
        zh: "<strong>真实航拍 Mip-NeRF 360 garden</strong> · 583 万全 SH3 高斯 · 片内 VMEM 原生溅射"
      },
      hudR: {
        en: "Native <strong>4K @ 2.61 FPS</strong> (<strong>8K @ 29.19 MPix/s</strong>) · Click to Inspect",
        zh: "原生 <strong>4K 2.61 FPS</strong>（<strong>8K 29.19 MPix/s</strong>）· 点击放大检视"
      }
    }
  ],

  compareLabs: {
    blender: {
      title: {
        en: "SLIDER 01 · TPU-RTX VS. BLENDER CYCLES GROUND TRUTH (6 STUDIO SCENES)",
        zh: "拉杆对比 01 · TPU-RTX 与官方 BLENDER CYCLES 真值同机位实时对照（6 大电影级场景）"
      },
      presets: [
        {
          id: "bmw",
          tab: { en: "BMW27 (37.65 dB)", zh: "BMW27 跑车 (37.65 dB)" },
          aImg: "./assets/compare/bmw_a.webp",
          bImg: "./assets/compare/bmw_b.webp",
          aLbl: { en: "LEFT: TPU-RTX Wide-BVH-64 + AgX (0.9813 SSIM)", zh: "左：TPU-RTX Wide-BVH-64 + AgX (SSIM 0.9813)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        },
        {
          id: "barcelona",
          tab: { en: "Barcelona 54.9M", zh: "巴塞罗那馆 5,491 万面" },
          aImg: "./assets/compare/barcelona_a.webp",
          bImg: "./assets/compare/barcelona_b.webp",
          aLbl: { en: "LEFT: TPU-RTX (54.91M Tris, 15.94 mm Z-MAE)", zh: "左：TPU-RTX (5,491 万面, 深度差 15.94 mm)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        },
        {
          id: "monster",
          tab: { en: "Monster SSS (36.98 dB)", zh: "床下怪兽 SSS (36.98 dB)" },
          aImg: "./assets/compare/monster_a.webp",
          bImg: "./assets/compare/monster_b.webp",
          aLbl: { en: "LEFT: TPU-RTX 1024² (36.98 dB, 0.121 mm Z-MAE)", zh: "左：TPU-RTX 1024² (36.98 dB, 深度差 0.121 mm)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        },
        {
          id: "fishycat",
          tab: { en: "Fishy Cat Fur (28.80 dB)", zh: "Fishy Cat 毛发 (28.80 dB)" },
          aImg: "./assets/compare/fishycat_a.webp",
          bImg: "./assets/compare/fishycat_b.webp",
          aLbl: { en: "LEFT: TPU-RTX (7,255 Fur Instances, 0.9592 SSIM)", zh: "左：TPU-RTX (7,255 毛发实例, SSIM 0.9592)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        },
        {
          id: "barbershop",
          tab: { en: "Barbershop 141-Tex", zh: "理发店 141 张贴图" },
          aImg: "./assets/compare/barbershop_a.webp",
          bImg: "./assets/compare/barbershop_b.webp",
          aLbl: { en: "LEFT: TPU-RTX (8.55M Tris, 31.00 dB Orbit)", zh: "左：TPU-RTX (855 万面, 31.00 dB 环绕)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        },
        {
          id: "lonemonk",
          tab: { en: "Lone Monk 20.1M", zh: "独行僧侣 2,014 万面" },
          aImg: "./assets/compare/lonemonk_a.webp",
          bImg: "./assets/compare/lonemonk_b.webp",
          aLbl: { en: "LEFT: TPU-RTX (28.83 dB, 18.91 mm Z-MAE)", zh: "左：TPU-RTX (28.83 dB, 深度差 18.91 mm)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        }
      ]
    },
    optics: {
      title: {
        en: "SLIDER 02 · PHYSICAL OPTICS, 4K BOKEH, ReSTIR DI & DEFORMABLE CLOTH (8 PRESETS)",
        zh: "拉杆对比 02 · 波动薄膜、无限镜廊、4K 景深、ReSTIR 与实时可变形布料（8 组效果）"
      },
      presets: [
        {
          id: "thinfilm",
          tab: { en: "Thin-Film + Aniso GGX", zh: "Airy 薄膜 + 拉丝金属" },
          aImg: "./assets/compare/thinfilm_a.webp",
          bImg: "./assets/compare/thinfilm_b.webp",
          aLbl: { en: "LEFT: Isotropic Baseline (No Interference)", zh: "左：常规各向同性基线（无干涉）" },
          bLbl: { en: "RIGHT: 8-Bin Airy Thin-Film + Radial Aniso GGX", zh: "右：8 波段 Airy 薄膜干涉 + 径向拉丝 GGX" }
        },
        {
          id: "mirror",
          tab: { en: "2 vs 12-Bounce Mirrors", zh: "2 次 vs 12 次无限镜廊" },
          aImg: "./assets/compare/mirror_a.webp",
          bImg: "./assets/compare/mirror_b.webp",
          aLbl: { en: "LEFT: Shallow max_bounces = 2 (Black Tunnel)", zh: "左：浅层 2 次反弹（隧道提前截断发黑）" },
          bLbl: { en: "RIGHT: 12-Bounce Infinite Recursion while_loop", zh: "右：单 while_loop 12 次深度镜廊递归" }
        },
        {
          id: "bokeh",
          tab: { en: "4K f/1.4 Near vs Far Bokeh", zh: "4K f/1.4 前景/远景对焦" },
          aImg: "./assets/compare/bokeh_a.webp",
          bImg: "./assets/compare/bokeh_b.webp",
          aLbl: { en: "LEFT: Native 4K 55mm f/1.4 Near-Field Focus (108.7 ms)", zh: "左：原生 4K 55mm f/1.4 近景对焦（108.7 ms）" },
          bLbl: { en: "RIGHT: Native 4K 55mm f/1.4 Far-Field Focus Pull", zh: "右：原生 4K 55mm f/1.4 远景拉焦散景" }
        },
        {
          id: "dispersion",
          tab: { en: "Cauchy Prism Dispersion", zh: "柯西棱镜光谱色散" },
          aImg: "./assets/compare/dispersion_a.webp",
          bImg: "./assets/compare/dispersion_b.webp",
          aLbl: { en: "LEFT: Achromatic Crown Glass (Fixed IOR)", zh: "左：常规无色散玻璃（固定折射率）" },
          bLbl: { en: "RIGHT: 6-Band Cauchy Spectral Dispersion", zh: "右：6 波段柯西分波长彩虹光谱色散" }
        },
        {
          id: "sss",
          tab: { en: "Jade Random-Walk SSS", zh: "玉石随机游走 SSS" },
          aImg: "./assets/compare/sss_a.webp",
          bImg: "./assets/compare/sss_b.webp",
          aLbl: { en: "LEFT: Opaque Dielectric Shader", zh: "左：不透明硬表面对照材质" },
          bLbl: { en: "RIGHT: 16-Step Random-Walk Volumetric Jade SSS", zh: "右：16 步体内随机游走翡翠次表面散射" }
        },
        {
          id: "godrays",
          tab: { en: "Cathedral God Rays", zh: "教堂体积丁达尔光" },
          aImg: "./assets/compare/godrays_a.webp",
          bImg: "./assets/compare/godrays_b.webp",
          aLbl: { en: "LEFT: Clean Vacuum Transport (No Participating Media)", zh: "左：无参与介质真空光线传输" },
          bLbl: { en: "RIGHT: BVH-Shadowed Volumetric Scattering Shafts", zh: "右：BVH 阴影体积散射教堂圣光束" }
        },
        {
          id: "restir",
          tab: { en: "1,024-Light ReSTIR DI", zh: "1,024 盏灯 ReSTIR DI" },
          aImg: "./assets/compare/restir_a.webp",
          bImg: "./assets/compare/restir_b.webp",
          aLbl: { en: "LEFT: Equal-Ray Naive 2-Vis MIS (10.29 dB)", zh: "左：等预算朴素 MIS 采样 (10.29 dB)" },
          bLbl: { en: "RIGHT: TPU-ReSTIR DI + Denoise (21.89 dB, 7.43x VarRed)", zh: "右：TPU-ReSTIR DI + 降噪 (21.89 dB, 方差降 7.43 倍)" }
        },
        {
          id: "cloth",
          tab: { en: "Deformable Cloth BVH Refit", zh: "动态布料 BVH Refit (973 FPS)" },
          aImg: "./assets/compare/cloth_a.webp",
          bImg: "./assets/compare/cloth_b.webp",
          aLbl: { en: "LEFT: Frame t=0 Cloth Release (32,768 Dynamic Tris)", zh: "左：t=0 悬空布料释放（32,768 动态面）" },
          bLbl: { en: "RIGHT: Frame t=96 Draped Sphere (973 FPS Refit, 0.0 m MAE)", zh: "右：t=96 悬垂包裹球体（973 FPS 零误差 Refit）" }
        }
      ]
    },
    neural: {
      title: {
        en: "SLIDER 03 · 4DGS CONTACT, TPU-DLSS 3.5, 4K DENOISER & ZERO-RAYTRACE RELIGHT (5 PRESETS)",
        zh: "拉杆对比 03 · 4DGS 物理推挤、TPU-DLSS 3.5、4K 降噪与零光追重打光（5 组效果）"
      },
      presets: [
        {
          id: "robowild4d",
          tab: { en: "4DGS x MJX Push (+16.4 cm)", zh: "4DGS x MJX 推挤 (+16.4 cm)" },
          aImg: "./assets/compare/robowild4d_a.webp",
          bImg: "./assets/compare/robowild4d_b.webp",
          aLbl: { en: "LEFT: t=0 Before MJX Panda Contact (125.6k LEGO Splats)", zh: "左：t=0 撞击前（kitchen 中 12.56 万乐高高斯）" },
          bLbl: { en: "RIGHT: After MJX Push (+16.43 cm / -17.71 deg, 5.32 FPS 1080p)", zh: "右：Franka 实臂推走 +16.43 cm / -17.71°（5.32 FPS）" }
        },
        {
          id: "dlss",
          tab: { en: "TPU-DLSS 3.5 (1080p -> 4K)", zh: "TPU-DLSS 3.5 (58.9 FPS 4K)" },
          aImg: "./assets/compare/dlss_a.webp",
          bImg: "./assets/compare/dlss_b.webp",
          aLbl: { en: "LEFT: Noisy 1080p 1-spp Input Frame", zh: "左：1080p 1-spp 原始噪点输入帧" },
          bLbl: { en: "RIGHT: Native 4K TPU-DLSS 3.5 Output (58.93 FPS, 3.91x Speedup)", zh: "右：原生 4K TPU-DLSS 3.5 输出（58.93 FPS，3.91 倍提速）" }
        },
        {
          id: "denoise",
          tab: { en: "1-spp -> 4K MXU Denoiser", zh: "1-spp 噪点 -> 4K 降噪 (87.8 FPS)" },
          aImg: "./assets/compare/denoise_a.webp",
          bImg: "./assets/compare/denoise_b.webp",
          aLbl: { en: "LEFT: Raw 1-spp 4-Bounce Monte Carlo (9.70 dB)", zh: "左：原始 1-spp 4 次反弹路径追踪噪点 (9.70 dB)" },
          bLbl: { en: "RIGHT: BF16 MXU 4K Denoiser (24.92 dB, 11.39 ms / 87.8 FPS)", zh: "右：BF16 MXU 4K 降噪输出 (24.92 dB, 11.39 ms)" }
        },
        {
          id: "material",
          tab: { en: "Matte Clay -> Chrome (0 RT)", zh: "哑光泥塑 -> 镜面铬合金 (0 光追)" },
          aImg: "./assets/compare/material_a.webp",
          bImg: "./assets/compare/material_b.webp",
          aLbl: { en: "LEFT: Matte Clay (rough=0.88, metal=0.00)", zh: "左：哑光泥塑材质（零额外光追）" },
          bLbl: { en: "RIGHT: Brushed Chrome (rough=0.03, metal=1.00, 78.6 FPS 4K)", zh: "右：高光铬合金（原生 4K BF16 实时 78.6 FPS）" }
        },
        {
          id: "tod",
          tab: { en: "Sunset 2600K -> Twilight 9500K", zh: "日落 2600K -> 暮蓝 9500K" },
          aImg: "./assets/compare/tod_a.webp",
          bImg: "./assets/compare/tod_b.webp",
          aLbl: { en: "LEFT: Golden Sunset (2600K, az=202 deg)", zh: "左：黄金日落暖光 (2600K, 静态 G-Buffer)" },
          bLbl: { en: "RIGHT: Blue-Hour Twilight (9500K, 307 FPS 1080p / 78.6 FPS 4K)", zh: "右：蓝调暮光重照明（1080p 307 FPS / 4K 78.6 FPS）" }
        }
      ]
    }
  },

  benchmarks: {
    mjx: {
      title: {
        en: "Embodied Robotics, Isaac Replicator DR, Menagerie, Tactile & Sim-to-VLA (host_callbacks=0)",
        zh: "具身机器人、Isaac 域随机、Menagerie、GelSight 视触觉与 Sim-to-VLA（零回调）"
      },
      note: {
        en: "Single-XLA-program physics + rendering + domain randomization + VLA training directly in TPU HBM.",
        zh: "物理仿真、光线追踪、域随机化与策略网络前/反向传播全部驻留 TPU HBM 单图执行。"
      },
      rows: [
        { label: { en: "V3 Fused 4c MJX+Render (64x64, B=8192)", zh: "V3 融合 4 芯 MJX+渲染 (64x64, B=8192)" }, val: 266209, unit: "FPS E2E", pct: 98, extra: { en: "921,309 render-equiv", zh: "921,309 纯渲染等效" } },
        { label: { en: "4-Sensor Visuo-Tactile GelSight+Cams (4c)", zh: "4 路视触觉 GelSight+双目相机 (4 芯)" }, val: 270625, unit: "sensor-FPS", pct: 100, extra: { en: "67,656 env-steps/s (0.15% DR)", zh: "67,656 env/s (DR 仅 +0.15%)" } },
        { label: { en: "Isaac Replicator RoboTwin 3-Cam+DR (4c)", zh: "Isaac Replicator RoboTwin 三摄+DR (4 芯)" }, val: 123858, unit: "cam-FPS", pct: 46, extra: { en: "41,286 env/s (+0.32% DR)", zh: "41,286 env/s (DR +0.32%)" } },
        { label: { en: "V3 Closed-Loop Vision PPO Cartpole (4c)", zh: "V3 零拷贝闭环 Vision PPO Cartpole (4 芯)" }, val: 656358, unit: "env-steps/s", pct: 95, extra: { en: "steady · Panda 211.6k / Humanoid 292.0k", zh: "稳态 · Panda 211.6k / Humanoid 292.0k" } },
        { label: { en: "Factory G=2048 Sim-to-VLA ViT-Tiny (4c)", zh: "千帧工厂 G=2048 在线 Sim-to-VLA (4 芯)" }, val: 7143, unit: "frames/s", pct: 12, extra: { en: "1M frames/142.6s · -1.84% VLA", zh: "百万帧 142.6s · -1.84% 融合开销" } },
        { label: { en: "MuJoCo Menagerie Wide-BVH-64 (Aero/Panda/G1)", zh: "MuJoCo Menagerie Wide-BVH-64 (4 芯)" }, val: 4284, unit: "FPS (64²)", pct: 8, extra: { en: "3,131 Panda / 2,546 G1 / 1,863 Go2", zh: "Panda 3,131 / G1 2,546 / Go2 1,863" } },
        { label: { en: "End-to-End Diff MJX+PBR Servoing (w0)", zh: "端到端可微 MJX+光追 像素到力矩伺服" }, val: 27.9, unit: "grad-steps/s", pct: 4, extra: { en: "36.08 dB · 1.63 cm EE err", zh: "36.08 dB · 末端误差 1.63 cm" } }
      ]
    },
    gs3d: {
      title: {
        en: "Native 3DGS / 4DGS x MJX & 300M-Splat 4-Chip HBM Scaling (gather_ops=0)",
        zh: "原生 3DGS / 4DGS x MJX 与单机 4 芯 3 亿高斯容量极限（gather_ops=0）"
      },
      note: {
        en: "Zero dynamic HBM gathers; scales from 142k FPS (N=2048) to 300M Gaussians (79.44% of 4-chip 380 GB HBM) and 8K photogrammetry.",
        zh: "无任何动态 HBM Gather；吞吐高达 14.2 万 FPS，HBM 容量推至 3 亿个高斯（占 4 芯 380 GB 显存 79.44%）及原生 8K 实景渲染。"
      },
      rows: [
        { label: { en: "V2 jax_tile Baseline (1c 64x64, N=2048)", zh: "V2 jax_tile 基线 (单芯 64x64, N=2048)" }, val: 1194, unit: "FPS", pct: 2, extra: { en: "baseline (Pallas was 659)", zh: "基线 (初版 Pallas 仅 659)" }, base: true },
        { label: { en: "V3 gs3d_v3 BF16-MXU (1c 64x64, N=2048)", zh: "V3 gs3d_v3 BF16-MXU (单芯 64x64)" }, val: 35916, unit: "FPS", pct: 25, extra: { en: "30.1x V2", zh: "30.1x 提速" } },
        { label: { en: "V3 gs3d_v3 BF16-MXU (4c shard_map, 64x64)", zh: "V3 gs3d_v3 4 芯扩展 (64x64)" }, val: 142008, unit: "FPS", pct: 100, extra: { en: "155,858 FPS @ 128² dry-run", zh: "128² 验证 155,858 FPS" } },
        { label: { en: "Fast 4c 32-View jax.grad Training (garden)", zh: "4 芯 32 视角 jax.grad 快速训练 (garden)" }, val: 749, unit: "view-steps/s", pct: 18, extra: { en: "23.4 steps/s -> 31.3 dB peak", zh: "23.4 steps/s · 峰值 31.3 dB" } },
        { label: { en: "Mip-NeRF 360 4K Quartet (bonsai/kitchen/garden)", zh: "Mip-NeRF 360 原生 4K 实景 (SH3)" }, val: 4.20, unit: "FPS @ 4K", pct: 12, extra: { en: "8K garden 29.19 MPix/s", zh: "8K garden 29.19 MPix/s" } },
        { label: { en: "4DGS x MJX Dynamic Push (125.6k LEGO splats)", zh: "4DGS x MJX 动态推挤 (12.5 万高斯)" }, val: 5.32, unit: "FPS @ 1080p", pct: 14, extra: { en: "1.30 FPS @ 4K (+16.4 cm push)", zh: "4K 1.30 FPS (推挤 +16.4 cm)" } },
        { label: { en: "Single-Host 300,000,000 Gaussians @ 4K (4c)", zh: "单机 4 芯 300,000,000 高斯 @ 4K" }, val: 301.87, unit: "GB HBM", pct: 80, extra: { en: "0.25 FPS · 1,171,875:1 cull", zh: "0.25 FPS · 剔除比 117 万:1" } }
      ]
    },
    cinema: {
      title: {
        en: "Disney Moana OpenUSD v2.1 (22.30B Tris), Deformable Cloth & Blender Studio on TPU v5p",
        zh: "迪士尼 Moana OpenUSD (223 亿面)、可变形布料与 Blender Studio 电影标准场景"
      },
      note: {
        en: "Compiled directly from official OpenUSD and .blend files into Wide-BVH-64 packs + AgX tonemapping.",
        zh: "直接由官方 .usd 与 .blend 原生编译为 Wide-BVH-64 与 Cycles 材质图 + AgX 色调映射。"
      },
      rows: [
        { label: { en: "BMW27 Clearcoat (Orbit PSNR)", zh: "BMW27 双层清漆跑车 (环绕 PSNR)" }, val: 37.65, unit: "dB PSNR", pct: 100, extra: { en: "0.9813 SSIM · -0.012% lum", zh: "SSIM 0.9813 · 亮度差 -0.012%" } },
        { label: { en: "Monster Under the Bed SSS 1024² (Orbit)", zh: "床下怪兽 SSS 1024² (环绕 PSNR)" }, val: 36.98, unit: "dB PSNR", pct: 98, extra: { en: "35.34 dB hero · 0.121 mm Z-MAE", zh: "特写 35.34 dB · 深度差 0.121 mm" } },
        { label: { en: "Barbershop Interior 8.55M Tris (141 Tex)", zh: "理发店室内 855 万面 (141 张纹理)" }, val: 31.00, unit: "dB PSNR", pct: 82, extra: { en: "13.8 mm MedAE · 4K 1.16 img/s", zh: "中位深度 13.8 mm · 4K 1.16 img/s" } },
        { label: { en: "Junkshop / Lone Monk / Fishy Cat (Orbit)", zh: "Junkshop / Lone Monk / Fishy Cat" }, val: 30.36, unit: "dB PSNR", pct: 80, extra: { en: "28.83 Monk / 28.80 Fishy Cat", zh: "Monk 28.83 / Fishy Cat 28.80 dB" } },
        { label: { en: "Barcelona Pavilion (54.91M Unique Tris)", zh: "巴塞罗那展馆 (5,491 万唯一三角面)" }, val: 27.45, unit: "dB PSNR", pct: 73, extra: { en: "15.94 mm Z-MAE · 4K 1.11 img/s", zh: "深度差 15.94 mm · 4K 1.11 img/s" } },
        { label: { en: "Deformable Cloth BVH Refit (32,768 tris)", zh: "可变形布料实时 BVH Refit (3.28 万面)" }, val: 973.3, unit: "FPS (64²)", pct: 88, extra: { en: "0.0 m MAE · 20.39 FPS @ 1024²", zh: "0.0 m 误差 · 1024² 20.39 FPS" } },
        { label: { en: "Disney Moana OpenUSD (22.30B Tris, 4K)", zh: "迪士尼 Moana OpenUSD (223 亿面, 4K)" }, val: 5130, unit: "ms / frame", pct: 60, extra: { en: "8K 17.2s · Underwater 8.7s", zh: "8K 17.2s · 水下珊瑚礁 8.7s" } }
      ]
    },
    dlss: {
      title: {
        en: "TPU-DLSS 3.5, 4K MC Denoiser & Zero-Copy Static G-Buffer Relighting (4-Chip MXU)",
        zh: "TPU-DLSS 3.5、4K 路径追踪降噪与零光追交互式重打光（4 芯 MXU）"
      },
      note: {
        en: "Exact 4x4 bilinear reprojection + optimization-barrier BF16/INT8 MXU inference at 1080p and Native 4K.",
        zh: "精确 4x4 双线性重投影 + optimization_barrier BF16/INT8 MXU 推理。"
      },
      rows: [
        { label: { en: "Interactive Static G-Buffer Relight (4c 1080p BF16)", zh: "零光追静态 G-Buffer 交互重打光 (1080p BF16)" }, val: 307.4, unit: "FPS", pct: 100, extra: { en: "20.28 dB · 0 raytrace ops", zh: "20.28 dB · 0 次光追开销" } },
        { label: { en: "4K BF16 MXU MC AI Denoiser (4c 3840x2160)", zh: "原生 4K BF16 MXU 蒙特卡洛降噪器 (4 芯)" }, val: 87.8, unit: "FPS @ 4K", pct: 35, extra: { en: "11.39 ms · +15.22 dB (1-spp)", zh: "11.39 ms · +15.22 dB (1-spp)" } },
        { label: { en: "Interactive Relight Native 4K BF16 (vs 37.2 FP32)", zh: "原生 4K BF16 交互重打光 (对比 FP32 37.2)" }, val: 78.6, unit: "FPS @ 4K", pct: 31, extra: { en: "2.11x FP32 · INT8 78.1 FPS", zh: "提速 2.11x · INT8 78.1 FPS" } },
        { label: { en: "TPU-DLSS 3.5 Native 4K (from 1080p 1-spp, 4c)", zh: "TPU-DLSS 3.5 原生 4K 重建 (1080p->4K, 4 芯)" }, val: 58.9, unit: "FPS @ 4K", pct: 25, extra: { en: "16.97 ms · 3.91x E2E speedup", zh: "16.97 ms · 端到端提速 3.91x" } }
      ]
    },
    fx: {
      title: {
        en: "TPU-Native Monte Carlo GI, ReSTIR DI (1,024 Lights), Thin-Lens Bokeh & Physical FX",
        zh: "原生蒙特卡洛全局光、ReSTIR DI（1,024 盏灯）、薄透镜散景与波动光学"
      },
      note: {
        en: "100% pure ray/path transport & wave optics compiled with zero host callbacks.",
        zh: "100% 纯射线/路径传输与波动光学，零主机回调直接跑在 TPU 上。"
      },
      rows: [
        { label: { en: "Native 4K 16-spp f/16 Thin-Lens Bokeh (4c)", zh: "原生 4K 16-spp f/16 薄透镜散景 (4 芯)" }, val: 18.28, unit: "FPS @ 4K", pct: 100, extra: { en: "54.7 ms · 32-spp f/1.4: 9.20 FPS", zh: "54.7 ms · 32-spp f/1.4 9.20 FPS" } },
        { label: { en: "3-Bounce Native 8K Path Tracer (4c Dry-Run)", zh: "3 次反弹原生 8K 路径追踪 (4 芯实测)" }, val: 218.9, unit: "spm @ 8K", pct: 68, extra: { en: "726.3 MRays/s · host_cb=0", zh: "726.3 MRays/s · 零回调" } },
        { label: { en: "1-Bounce / 3-Bounce Physical GI (512², 1c)", zh: "单芯 512² 1-bounce / 3-bounce 全局光" }, val: 3121, unit: "spm @ 512²", pct: 85, extra: { en: "835 spm 3-bounce (64 spp)", zh: "3-bounce 835 spm (64 spp)" } },
        { label: { en: "TPU-ReSTIR DI 1,024-Light Courtyard (720p, 4c)", zh: "TPU-ReSTIR DI 1,024 动态光源庭院 (720p)" }, val: 21.89, unit: "dB PSNR", pct: 60, extra: { en: "+11.61 dB · 7.43x HDR VarRed", zh: "+11.61 dB · HDR 方差降 7.43x" } },
        { label: { en: "Airy Thin-Film (8-bin) + 12-Bounce Mirrors (4c)", zh: "8 波段 Airy 薄膜 + 12 次递归镜廊 (4 芯)" }, val: 12, unit: "bounces", pct: 50, extra: { en: "128 spp · single XLA while_loop", zh: "128 spp · 单条 XLA while_loop" } }
      ]
    }
  },

  ledger: [
    { cat: "robotics", name: "V3 Fused MJX+Render E2E", scale: "4c, 64x64, B=8192", speed: "266,209 FPS (921k equiv)", acc: "depth_mae = 3.35e-10 m", file: "v3_w0_wave2_100k_fps.json" },
    { cat: "robotics", name: "Zero-Copy Vision PPO (Cartpole/Panda/Humanoid)", scale: "4c, 64x64, B=1024-4096", speed: "656.4k / 211.6k / 292.0k env/s", acc: "Cartpole eval 94.2% success", file: "v3_w0_wave4_rl_sdk.json" },
    { cat: "robotics", name: "Diff Physics + PBR Visual Servoing", scale: "1c, 12-step MJX, 7-DOF", speed: "27.87 grad-steps/s (8.61s)", acc: "36.08 dB · EE 16.81->1.63 cm", file: "v3_w0_taskB.json" },
    { cat: "robotics", name: "MuJoCo Menagerie Multi-Mesh Wide-BVH-64", scale: "4c, Aero/Panda/G1/Go2", speed: "4,284 / 3,131 / 2,546 / 1,863 FPS", acc: "host_callbacks=0, gather_ops=0", file: "v3_w0_taskA.json" },
    { cat: "robotics", name: "Isaac Replicator RoboTwin 3-Cam DR", scale: "4c, 5-modality DR", speed: "123,858 cam-FPS (+0.32% DR)", acc: "41,286 env-steps/s", file: "v3_w1_wave3_replicator.json" },
    { cat: "robotics", name: "1M Factory Frames + Sim-to-VLA + GelSight", scale: "4c, G=2048, ViT-Tiny 1.02M", speed: "1M/142.6s · VLA 7,143 · Tactile 270.6k/s", acc: "Loss 0.2378->0.0332 (-1.84% VLA)", file: "v3_w1_wave4_vla_tactile_factory.json" },
    { cat: "robotics", name: "Autonomous 8x1080p + 2x128x2048 LiDAR Rig", scale: "4c, B=16 fused XLA graph", speed: "8.35 env-FPS · 66.8 cam · 142.9 MRays/s", acc: "Dual-XYZ MAE = 3.13e-8 m", file: "v3_w6_wave5_surround_bokeh.json" },
    { cat: "film", name: "Disney Moana OpenUSD v2.1 (4K/8K)", scale: "4c, 22.30B tris, 4.65 GiB", speed: "4K 5.13 s · 8K 17.24 s", acc: "0 host callbacks, 0 gathers", file: "v3_w2_wave5_moana_8k_mjx.json" },
    { cat: "film", name: "Disney Moana Golden Sunset & Underwater Reef", scale: "4c, 22.30B tris, Native 4K", speed: "Sunset 5.60 s · Reef 8.72 s", acc: "16-step HG Volumetrics + Caustics", file: "v3_w2_wave6_moana_sunset_reef.json" },
    { cat: "film", name: "Deformable Cloth Live BVH Refit", scale: "1c, 32,768 dynamic tris", speed: "973.3 FPS (64²) · 20.39 FPS (1024²)", acc: "0.0 m Z-diff vs CPU BVH rebuild", file: "v3_w2_deformable_cloth.json" },
    { cat: "film", name: "Blender BMW27 & Barcelona Pavilion (54.9M)", scale: "4c, Cycles Graph -> AgX", speed: "4K 1.11 img/s (Barcelona)", acc: "BMW27 37.65 dB (0.9813 SSIM)", file: "v3_w5_wave5_bmw_barcelona.json" },
    { cat: "film", name: "Blender Monster SSS (1024²) & Fishy Cat", scale: "4c, 1.60M / 907k tris", speed: "10.07 img/s (1024²) · 1.72 4K", acc: "Monster 36.98 dB · 0.121 mm Z-MAE", file: "v3_w5_wave6_studio_double.json" },
    { cat: "film", name: "Blender Barbershop Interior (141 Textures)", scale: "4c, 8.55M tris, 249 mat", speed: "4K 1.16 img/s", acc: "31.00 dB orbit · 13.8 mm MedAE", file: "v3_w7_barbershop_cinema.json" },
    { cat: "gs3d", name: "Native TPU 3DGS Rasterizer (gs3d_v3)", scale: "4c, 64x64, gather_ops=0", speed: "142,008 FPS (30.1x on 1c)", acc: "gather_ops = 0 in XLA HLO", file: "v3_w3_gs3d_v3.json" },
    { cat: "gs3d", name: "300M Gaussians in 4c HBM + 4DGSxMJX Push", scale: "4c, 301.87 GB / 380 GB HBM", speed: "4K 0.25 FPS (300M) · 4DGS 5.32 FPS", acc: "1,171,875:1 cull · +16.43 cm push", file: "v3_w3_wave6_4dgs_300m.json" },
    { cat: "gs3d", name: "Mip-NeRF 360 garden Native 8K & 32-View Grad", scale: "4c, 5.83M SH3 / 50k train", speed: "8K 0.88 FPS (1.14s) · 23.4 steps/s", acc: "20.28 -> 31.31 dB peak in 17.1 s", file: "v3_w3_wave7_8k_and_gradtrain.json" },
    { cat: "neural", name: "TPU-DLSS 3.5 (1080p 1-spp -> Native 4K)", scale: "4c BF16 MXU, 4x4 exact", speed: "58.93 FPS @ 4K (16.97 ms)", acc: "3.91x E2E speedup · 7.01x flicker cut", file: "v3_w4_wave4_tpu_dlss_4k.json" },
    { cat: "neural", name: "4K MC AI Denoiser & Zero-RT Relight Pareto", scale: "4c FP32 / BF16 / INT8 W8A8", speed: "87.78 FPS Denoise · 78.60 FPS Relight 4K", acc: "BF16 20.276 dB (2.11x) · INT8 -0.047 dB", file: "v3_w4_wave5_relight_quant.json" },
    { cat: "neural", name: "TPU-ReSTIR DI (1,024 Lights) + Airy/Mirrors", scale: "4c, 720p ReSTIR / 12-bounce", speed: "12-bounce while_loop (128 spp)", acc: "ReSTIR 21.89 dB (+11.61 dB, 7.43x VarRed)", file: "v3_w7_wave5_restir_cinema.json" },
    { cat: "neural", name: "4-Chip Pod Capstone Ceiling Dry-Run", scale: "4c (383 GB HBM, 5 suites)", speed: "2.15B tris · 8.25B rays/s LiDAR · 156k 3DGS", acc: "8K 3-bounce 218.9 spm · 0 host_cb", file: "v3_w6_pod_capstone_dryrun4c.json" },
    { cat: "neural", name: "32-Chip TPU v5p-64 Pod Capstone (8 Hosts)", scale: "32c (3.064 TB HBM, 395.1 s)", speed: "411k E2E · 952k 3DGS · 63.88B LiDAR · 1,650 spm 8K", acc: "17.18B unique tris · 1.062B splats · 0 CB", file: "v3_pod_v5p64_capstone.json" }
  ],

  copy: {
    en: {
      pageTitle: "TPU-RTX — Native Ray Tracing, Path Tracing & 3DGS on Google TPU | Feiyang Liu",
      themeNight: "Obsidian",
      themeDay: "Paper",
      nav: {
        c1: "01 GPU vs TPU", c2: "02 Evolution", c3: "03 Film & Optics", c4: "04 3DGS & Neural",
        c5: "05 Training & AI", c6: "06 Post-Mortems", c7: "07 Ledger & Index", c8: "08 Limits"
      },
      hero: {
        kicker: "GOOGLE CLOUD TPU v5p (1c TO 32-CHIP POD) · ZERO HOST_CALLBACK · ZERO GATHER_OPS",
        by: "By",
        license: "Apache-2.0",
        title: "<em>TPU-RTX</em>: Turning Google TPUs into Real&#8209;Time Ray&#8209;Tracing Engines.",
        lead: "Native <strong>Google Cloud TPU v5p</strong> ray tracing, path tracing, <strong>1.06B</strong>-splat 3DGS, and <strong>17.18B</strong>-triangle OpenUSD / Physical AI — compiled into one zero-copy XLA graph with zero CPU round-trips.",
        ctaStart: "GPU vs TPU Architecture",
        ctaGallery: "Ledger & Plate Index",
        stamp: "Pure Single-Scene Cinema Monograph · Zero Cropped Frames · Zero Multi-Panel Collages",
        t0i: "SCENE 01 · OPENUSD SUNSET",
        t0t: "Disney Moana Lagoon Sunset (22.30B Tris, 4K)",
        t1i: "SCENE 02 · FILM × MJX",
        t1t: "Live MJX Robots on Moana Beach (Native 4K)",
        t2i: "SCENE 03 · REAL-WORLD 3DGS",
        t2t: "Mip-NeRF 360 Garden @ Native 4K (SH3, 0 Gather)"
      },
      kpis: [
        ["4c Fused & 32c MJX+RTX", "266k / 411k FPS", "<strong>921,309 FPS</strong> render-equiv (4c) · <strong>411,023 FPS</strong> (32c E2E)"],
        ["Disney Moana OpenUSD", "22.30B Tris", "<strong>4K 5.13 s · 8K 17.24 s</strong> · Sunset &amp; Reef (4c, 4.65 GiB)"],
        ["3DGS Speed & Pod HBM", "952k FPS / 1.06B", "<code>142k</code> (4c) / <strong>952,248 FPS</strong> (32c 128²) · <strong>1.062B</strong> splats"],
        ["Native 8K Real 3DGS", "7680×4320", "<code>5.83M</code> SH3 <code>garden</code> @ <strong>29.19 MPix/s</strong> · 8K PT <strong>1,650 spm</strong> (32c)"],
        ["DLSS 3.5 & Relight (4K)", "58.9 / 78.6 FPS", "DLSS <strong>3.91×</strong> · Static G-Buf Relight <strong>78.6 FPS</strong> BF16 (4c)"],
        ["Blender Cycles Parity", "37.65 / 36.98 dB", "<code>BMW27</code> <strong>0.9813</strong> SSIM · <code>Monster</code> <strong>0.121 mm</strong> (4c)"],
        ["Cloth Refit & 32c Unique", "973 FPS / 17.18B", "<code>973.3 FPS</code> cloth BVH · <strong>17.18B</strong> unique tris (32c, 1.18 TB)"],
        ["Diff Phys + 32c LiDAR", "1.63 cm / 63.9B/s", "<code>jax.grad</code> <strong>36.08 dB</strong> (1c) · <strong>63.88B</strong> LiDAR rays/s (32c)"]
      ],
      c1: {
        num: "01 / 08 · GPU VS TPU",
        rail: "First principles: how SIMT RTX GPUs and Systolic TPU MXUs execute ray tracing differently — and where each wins or loses.",
        title: "Why <em>TPU</em>? GPU vs. TPU Silicon &amp; Tradeoffs.",
        lead: "A GPU (SIMT Warps + Hardware RT Cores + L1/L2 Caches) and a Google Cloud TPU v5p (<strong>128×128 Systolic MXU</strong> at 459 TFLOP/s BF16 + <strong>64 MiB software-managed VMEM</strong> + <strong>95.74 GiB HBM</strong>) sit on opposite ends of silicon architecture. Writing CUDA-style pointer chasing stalls on TPU gather latency; aligning traversal with 64-ary VMEM vectors turns 383 GB/host HBM and MXU throughput into a superpower.",
        gvtTag: "SILICON & DATAFLOW COMPARISON",
        gvtTitle: "SIMT RTX GPU vs. Systolic Google TPU v5p Execution Pipeline",
        gvtSub: "Why CUDA-style pointer chasing stalls on TPU HBM, and how VMEM 64-ary Wide-BVH unlocks systolic ray tracing.",
        gvtRows: [
          ["1. Dynamic Branching & Hardware RT Cores", "<span class='badge fail'>Con</span> No HW RT Cores; <code>128</code>-lane VPU prefers regular, branchless masking.", "<span class='badge ok'>Pro</span> Dedicated hardware RT Cores &amp; 32-thread SIMT warps handle incoherent multi-bounce branching naturally."],
          ["2. Random Memory Gather / Scatter", "<span class='badge fail'>Con</span> High <code>166 FLOP/B</code> ridge point; uncoalesced HBM gathers or tile atomics crater speed.", "<span class='badge ok'>Pro</span> Hardware L1/L2 caches and fast shared-memory atomics absorb irregular pointer chasing."],
          ["3. On-Chip VMEM & HBM Capacity", "<span class='badge ok'>Pro</span> Massive <strong>95.74 GiB HBM/chip (383 GB/host)</strong> + explicit 64 MiB VMEM holds <strong>300M splats</strong> or <strong>22.3B</strong>-tri Moana without PCIe paging.", "<span class='badge fail'>Con</span> Per-GPU VRAM (24–80 GB) caps massive 3DGS/OpenUSD + RL batch concurrency on one node."],
          ["4. Zero-Copy Physics + Render + RL/VLA", "<span class='badge ok'>Pro</span> Physics (<code>mjx.step</code>), raytrace, GelSight, LiDAR, PPO &amp; ViT-Tiny live in <strong>one XLA graph</strong> (<code>266k FPS</code>, <code>-1.84%</code> VLA overhead).", "<span class='badge fail'>Con</span> Often crosses Warp / OptiX / Isaac / PyTorch bridges or copies buffers between engines."],
          ["5. MXU AI Denoising, DLSS & Relighting", "<span class='badge ok'>Pro</span> <code>459 TFLOP/s</code> BF16 MXUs sit directly behind HBM G-buffers: <strong>87.8 FPS</strong> 4K Denoise, <strong>58.9 FPS</strong> 4K DLSS, <strong>78.6 FPS</strong> 4K Relight.", "<span class='badge ok'>Pro</span> Tensor Cores + mature DLSS 3.5 / OptiX OIDN ecosystem work out of the box."],
          ["6. Startup Compile vs. Steady Throughput", "<span class='badge fail'>Con</span> XLA AOT + Mosaic lowering pays a <strong>4–130 s</strong> cold compile upfront before steady loops.", "<span class='badge ok'>Pro</span> Sub-second JIT / precompiled CUDA kernels deliver instant interactive viewport responsiveness."],
          ["7. Authoring & DCC Ecosystem", "<span class='badge fail'>Con</span> Compiles exported <code>.usd</code>, <code>.blend</code>, MJCF, and <code>.ply</code> ahead of training.", "<span class='badge ok'>Pro</span> Unmatched interactive GUI tooling (Omniverse, Blender Viewport, Unreal Engine)."]
        ],
        pillars: [
          ["VMEM Traversal", "1. 3-Level 64-Ary Wide-BVH", "Packing 64 child AABBs into (8, 128) vector tiles lets Pallas Mosaic test 64 children in single VPU ops via pltpu.async_copy leaf streaming — scaling cleanly to 22.30B-tri Disney Moana and 2.15B unique tris."],
          ["Zero-Gather Raster", "2. In-VMEM Kernel Fusion & EWA", "Fusing raygen, traversal, GGX/AO/shadows, and material deduplication avoids HBM G-buffer round-trips (921k FPS render-equiv), while chunked VMEM take_along_axis accelerates 3DGS to 142k FPS and 300M splats."],
          ["Native MXU + Grad", "3. BF16/INT8 MXUs & jax.grad Physics", "Keeping every pixel inside TPU HBM lets BF16 MXUs run 4K denoising at 87.8 FPS, TPU-DLSS 3.5 at 58.9 FPS 4K, static G-buffer relighting at 78.6 FPS 4K, and pixel-to-torque backprop through mjx.step."]
        ],
        diagCap: "Architectural invariant verified in XLA HLO across all modules: host_callbacks = 0, send_recv_ops = 0, gather_ops = 0.",
        sdkBar: "Quickstart SDK · 8 Ecosystem Adapters under tpu_mjx_render.adapters"
      },
      c2: {
        num: "02 / 08 · EVOLUTION & PROFILING",
        rail: "Chronological step-envelope optimization log (including negative runs), 3DGS pivot, 64 MiB VMEM cliff, MXU Pareto, and Hero vs. Held-Out Orbit PSNR.",
        title: "<em>587×</em> on TPU: Optimization Log &amp; Orbit Generalization.",
        lead: "Inspired by <code>wenhaochai/rtx-h100</code>, we plot our chronological step-envelope optimization log (including flat/negative runs), the <code>tile_size = 512</code> VMEM spill cliff (<code>-24%</code>), BF16/INT8 MXU Pareto, and Hero vs. Held-Out Orbit PSNR across Blender scenes."
      },
      c3: {
        num: "03 / 08 · FILM, CLOTH & OPTICS",
        rail: "Two interactive left/right sliders (6 Blender Cycles scenes + 8 optical/cloth effects) plus 5 full-width standalone 4K/8K Moana & Caustics plates.",
        title: "<em>Moana</em> (22.3B Tris), Cycles Parity, Cloth &amp; Wave Optics.",
        lead: "Zero stitched multi-panel collages: compare 6 official Blender Cycles references and 8 optical/cloth regimes using the interactive left/right curtains below, followed by 5 full-width standalone 4K/8K Moana &amp; Caustics renders (all uncropped)."
      },
      c4: {
        num: "04 / 08 · 3DGS & NEURAL SILICON",
        rail: "Interactive 4K Single-Scene Mip-NeRF 360 + Bistro switcher + 5-preset 4DGS / DLSS / Denoiser / Relight split slider.",
        title: "Native TPU <em>3DGS / 4DGS</em>, <code>TPU-DLSS 3.5</code> &amp; 4K Relighting.",
        lead: "Switch between five full-width standalone 4K 3DGS scenes below (click any frame to open 1:1 Lightbox Loupe), then drag the 5-preset slider to inspect <code>4DGS × MJX</code> pushing, <strong>TPU-DLSS 3.5</strong> (<code>58.9 FPS</code> 4K), <strong>87.8 FPS 4K Denoising</strong>, and <strong>78.6 FPS</strong> zero-raytrace relighting."
      },
      c5: {
        num: "05 / 08 · TRAINING & PHYSICAL AI",
        rail: "Four live TPU convergence & throughput plots + full-width MuJoCo Menagerie PBR robot stage (G1, Go2, Panda, Aero Hand).",
        title: "In-HBM Training: <em>Diff Physics</em>, Vision PPO &amp; Menagerie.",
        lead: "Physics steps, sensors, losses, and gradients train together in TPU HBM. Below are four exact full-width telemetry curves extracted from <code>out/metrics/*.json</code> plus an interactive full-width <strong>MuJoCo Menagerie</strong> PBR switcher (Unitree G1, Go2, Franka Panda, Aero Hand)."
      },
      c6: {
        num: "06 / 08 · POST-MORTEMS",
        rail: "Real bottlenecks, negative Pallas benchmarks, neural shader ablations, multi-host pitfalls, and full-width QA-rejected renders.",
        title: "TPU Pitfalls: <em>Post-Mortems</em> &amp; QA-Rejected Renders.",
        lead: "Publishing only final hero renders hides what actually makes TPU rendering hard. Here are four genuine bottlenecks &amp; negative runs from our engineering ledger — plus two uncropped renders that our <strong>Visual QA Gate rejected</strong> before publication.",
        pmCards: [
          ["FAIL -> FIX", "E10 / E11 / E12", "1. The Hidden 151 MB Checkerboard Gather", "Amdahl profiling revealed >90% of early Pallas frame time (286.5 ms) wasn't in the BVH kernel — it was a single XLA dynamic gather geom_xmat[b_idx, safe_gid] triggered by one checkerboard floor plane.", "Replacing that dynamic lookup with static iteration over checkerboard IDs slashed shading 4.4x (286.5 -> 64.9 ms), unlocking 12,843 FPS (1c) and 50,131 FPS (4c) with gather_ops = 0."],
          ["NEGATIVE RESULT", "E14 vs Wave 3", "2. Hand-Written Pallas 3DGS Lost 1.81x to Pure XLA", "Our initial MXU log-space Pallas 3DGS kernel (gs3d_pallas) ran at only 658.6 FPS — slower than plain XLA jax_tile (1,193.2 FPS) because padded (BT, 64, 128) outputs bloated writes 16x and emitted 65,536 tiny DMAs.", "We documented the negative result, kept jax_tile default, then redesigned chunked VMEM EWA sorting (gs3d_v3) to jump 119x to 142,008 FPS (4c) and 300M splats in HBM."],
          ["3-RUN FAILURE -> PASS", "E4 / E5 / E6a -> E6b", "3. Why Stage-2 Neural Relighting Failed Three Times First", "M1 pixel MLP failed soft penumbras (27.80 dB); narrow W32 U-Net failed gripper occlusion (10% mismatch); jnp.clip inside loss saturated gradients on step 1 leaving training stuck at 15.20 dB.", "Adding reflect(V, N) + Fresnel N dot V, dual 1x1 BRDF skip heads, unclipped loss + global norm clipping, and tight-grasp augmentation passed 8/8 locked gates at 38.35 dB / 0.9889 SSIM."],
          ["MULTI-HOST PITFALL -> SOLVED", "32-Chip Pod 395.1 s", "4. Non-Addressable MJX Arrays & SSH PID Lock Leaks", "In 8-host (32-chip) JAX distributed mode, mjx.put_model(m, impl='jax') with device=None created cross-host IFRT failures, and multiline SSH PID cleanup previously left /dev/vfio/3 locked.", "Patched device=local_devs[0] + single-line tr PID cleanup, then executed the live 32-chip (3.064 TB HBM) Capstone across all 8 hosts in 395.07 s: 411k E2E, 952k 3DGS, 1.062B splats, 63.88B LiDAR rays/s, 17.18B unique tris & 1,650 spm @ 8K, returning 8/8 IDLE."]
        ],
        ablationRows: [
          ["Locked Phase-0 Threshold", ">= 30.0 dB", ">= 28.0 dB", "<= 0.010", "<span class='badge cyan'>All 8 Required</span>"],
          ["Stage 1 Direct Exact Baseline", "13.83 dB", "—", "—", "Hard shadows, no indirect GI"],
          ["E4: M1_pixel_128 (1x1 MLP, 150 ep)", "27.80 dB", "23.52 dB", "0.0500 (7/140)", "<span class='badge fail'>FAIL</span> Zero spatial receptive field"],
          ["E5: M2_spatial_32 (Narrow W32)", "35.58 dB", "27.79 dB", "0.1000 (14/140)", "<span class='badge fail'>FAIL</span> Grasping contact blur"],
          ["E6a-3: M2_spatial_128 + jnp.clip Loss", "15.20 dB", "—", "—", "<span class='badge fail'>FAIL</span> Dead step-1 gradient saturation"],
          ["E6b: M2_spatial_128 (24-ch BRDF, Final)", "<strong>38.35 dB</strong>", "<strong>30.31 dB</strong>", "<strong>0.00714 (1/140)</strong>", "<span class='badge ok'>PASS (8/8)</span> SSIM 0.9889, Edge 0.157 px"]
        ],
        rejected: [
          {
            id: "rejected_openusd_clay",
            img: "./assets/postmortem/rejected_openusd_clay.webp",
            banner: "QA REJECTED · EARLY FLAT CLAY SHADING & LOW MATERIAL CONTRAST",
            title: "Rejected Output A: Early Untextured OpenUSD Stage Test (v3_w2_openusd_4k)",
            desc: "Caught by Visual QA Gate: before binding full Disney OpenUSD Ptex/material packs, diffuse shaders collapsed distant foliage and water reflections into flat clay. Replaced by the verified Sunset, Reef Caustics, and Moana x MJX 4K passes."
          },
          {
            id: "rejected_garden_seams",
            img: "./assets/postmortem/rejected_garden_seams.webp",
            banner: "QA REJECTED · VISIBLE 16x16 TILE SEAMS & BLACK BACKGROUND",
            title: "Rejected Output B: Early 32-View 3DGS Training Render (v3_w3_garden_32v_grad)",
            desc: "Caught by Visual QA Gate: local tile accumulation bounds clipped splats outside each 16x16 frustum bin, producing blocky grid seams and masked black borders. Excluded from showcase galleries; fixed fast-grad pipeline verified at 31.31 dB peak."
          }
        ]
      },
      c7: {
        num: "07 / 08 · TELEMETRY & LEDGER",
        rail: "Interactive subsystem bars, zero-overflow 21-row JSON telemetry ledger, and quick-jump directory for all 25 standalone renders (no tiny thumbnail clutter).",
        title: "Telemetry Console, <em>21 JSON Logs</em> &amp; Scene Index.",
        lead: "Every metric traces directly to raw JSON artifacts in <code>out/metrics/</code> (compact <code>JSON ↗</code> links keep zero horizontal overflow). Use the quick-jump directory below to inspect any of the 25 standalone uncropped plates in full 1:1 Lightbox resolution."
      },
      c8: {
        num: "08 / 08 · 32-CHIP POD & COLOPHON",
        rail: "Live 32-chip (8-host, 3.064 TB HBM) scaling vs. 4-chip baseline, honest hardware comparison citing wenhaochai/rtx-h100, Feiyang Liu colophon & BibTeX.",
        title: "<em>32-Chip TPU Pod</em> (3.06 TB HBM), Limits &amp; <code>rtx-h100</code>.",
        lead: "Verified live 8-host (32-chip, <code>3.064 TB</code> HBM) Pod Capstone measurements completed in <code>395.07 s</code> alongside the 4-chip baseline (<code>383 GB</code> HBM), plus honest GPU vs. TPU boundaries citing <code>wenhaochai/rtx-h100</code> and Feiyang Liu BibTeX.",
        honest: "Every number reported on this page is measured live on real TPU v5p silicon (1-chip, 4-chip single-host, or 32-chip 8-host distributed pod, all explicitly labeled with zero fabricated scaling claims). Where GPUs still lead — sub-second interactive viewport turnaround, incoherent multi-bounce branching on hardware RT Cores, and single-card raw Cycles throughput (Chai et al., 2026) — we state so plainly.",
        authorBio: "Systems & graphics engineer exploring compiler-native rendering, differentiable physics, and high-throughput Physical AI on custom ML accelerators.",
        ack: "Third-Party Asset & Work Acknowledgments: Walt Disney Animation Studios Moana Island Scene v2.1 (Non-Commercial License); Blender Open Data / Blender Studio (BMW27, Barcelona Pavilion, Barbershop, Classroom, Junkshop, Lone Monk, Monster Under the Bed, Fishy Cat — CC0 / CC-BY); Mip-NeRF 360 photogrammetry captures (Barron et al.); Google DeepMind mujoco_menagerie (Apache-2.0 / BSD); Chai et al. (2026) wenhaochai/rtx-h100 Cycles H100/RTX PRO 6000 reference."
      },
      podStats: [
        ["Bench 1 & 2 · Embodied Physics + 3DGS (4c -> 32c)", "411,023 FPS E2E  |  952,248 FPS 3DGS", "8x pod scaling from 51.5k MJX+RTX (4c) to 411.0k FPS (32c B=131k); 128² 3DGS hits 952.2k FPS plus 1.062B ICI splats @ 1080p (3.41 s)."],
        ["Bench 3 · 128x2048 Batch 360° LiDAR (32c)", "243,676 env-FPS  |  63.88 Billion Rays/s", "B=16,384 parallel 128x2048 LiDAR scans in 67.2 ms across 8 hosts (B=32,768: 229,060 env-FPS / 60.05B rays/s, gather_ops=0)."],
        ["Bench 4 · Geometry-Sharded Unique Triangles (32c)", "17.18B Unique Non-Instanced Tris (1.18 TB)", "32-chip ICI jax.lax.pmin scales 8x over 4c (2.15B -> 17.18B unique tris), rendering 4K in 4.46 s and 8K in 8.53 s (8.59B tris: 2.43 s 4K / 4.70 s 8K)."],
        ["Bench 5 · Spatial-Sharded Native 8K 3-Bounce PT (32c)", "1,650.2 spm @ 8K (5.475 GRays/s, 7.54x 4c)", "Unbiased 3-bounce Monte Carlo GI @ 7680x4320 (33.18M px): 145.4 ms @ 4 spp (6.88 FPS) and 37.5 ms @ 1 spp (26.69 FPS, host_callbacks=0)."]
      ],
      compRows: [
        ["Embodied Vision RL, VLA & Tactile", "<strong>Zero-Copy Single XLA Graph</strong>: <code>266k FPS</code> E2E, <code>1M</code> factory frames in <code>142.6s</code>, <code>270.6k/s</code> GelSight, <code>-1.84%</code> Sim-to-VLA overhead", "Static offline renderer (no interactive RL loop)", "Mature CUDA Warp/Isaac ecosystem on RTX/L40S/H100"],
        ["Real-to-Sim 3DGS / 4DGS + Physics", "<strong>Zero-Gather Native TPU</strong>: <code>142k FPS</code> (<code>64²</code>), <code>300M</code> splats @ 4K (4c HBM), <code>8K garden 29.2 MPix/s</code>, live <code>4DGS x MJX</code> push", "Cycles mesh + NRC only", "Requires separate CUDA gsplat + USD/PhysX bridge"],
        ["End-to-End Diff Physics + Render", "<strong>36.08 dB / 1.63 cm EE Error</strong> via <code>jax.value_and_grad(mjx.step + PBR)</code> at <code>27.9 grad-steps/s</code>", "Non-differentiable offline path tracer", "Separate Warp diff-sim / Nvdiffrast integration"],
        ["Film Production Path Tracing & Cycles", "<strong>Large-Scene OpenUSD & .blend Compiler</strong>: <em>Moana</em> (<code>22.30B</code> tris, 4.65 GiB), <code>BMW27 37.65 dB</code>, <code>Monster 36.98 dB</code>", "<strong>Speed Champion (wenhaochai/rtx-h100)</strong>: <code>55,148 spm</code> single H100 (7.8x Blender), <code>78,297 spm</code> RTX PRO 6000 OptiX (39.9 img/s NRC)", "Full interactive Omniverse / OptiX workstation viewport"],
        ["AI Relighting, Denoising & DLSS", "<strong>Native MXU BF16/INT8</strong>: 4K Denoise <code>87.8 FPS</code>, 4K DLSS <code>58.9 FPS</code>, Static Relight <code>78.6 FPS</code> 4K / <code>307 FPS</code> 1080p", "NRC-scatter <code>40.4 img/s</code> (32-spp bar, H100)", "Hardware Tensor Core DLSS 3.5 + Ray Reconstruction"]
      ],
      ui: {
        copy: "Copy", copied: "Copied", zoom4k: "Inspect ↗", loupe: "1:1 Pixel Loupe (L)", raw: "Open Full Image ↗",
        footerL: "TPU-RTX · Google Cloud TPU v5p (JAX + Pallas Mosaic · Zero host_callbacks · Zero HBM Gathers)",
        searchPlaceholder: "Search experiments, scenes, JSON...",
        b_mjx: "MJX & Robotics (w0/w1/w6)", b_gs3d: "3DGS / 4DGS / 300M (w3)", b_cinema: "Moana, Cloth & Blender (w2/w5/w7)", b_dlss: "TPU-DLSS 3.5 & Relight (w4)", b_fx: "Path Tracing, ReSTIR & Optics (w7/w6)",
        g_all: "All Standalone Plates (25)", g_film: "Moana & Blender Studio (10)", g_optics: "Physical Optics & Caustics (4)", g_gs3d: "Real-World 3DGS / Bistro (5)", g_robotics: "Menagerie PBR Robots (4)", g_rej: "QA-Rejected (2)",
        th1: "Workload", th2: "TPU-RTX (Google TPU v5p)", th3: "GPU Cycles (wenhaochai/rtx-h100)", th4: "NVIDIA Isaac Sim / OptiX",
        gvt1: "Dimension", gvt2: "Google Cloud TPU v5p (TPU-RTX)", gvt3: "NVIDIA GPU (RTX / H100 / Isaac)",
        ab1: "Stage-2 Neural Shader Ablation", ab2: "PSNR (dB)", ab3: "ROI Contact PSNR", ab4: "Occ Mismatch", ab5: "Gate & Root Cause",
        ld_all: "All (21)", ld_rob: "Robotics", ld_film: "Film & Cloth", ld_gs: "3DGS/4DGS", ld_nn: "Neural/Optics",
        lt1: "Experiment", lt2: "Config / Scale", lt3: "Throughput", lt4: "Accuracy / Parity", lt5: "JSON",
        ev_t: "CHRONOLOGICAL STEP-ENVELOPE · NEGATIVE RUNS EXPOSED", ev_h: "587x Optimization Trajectory (453 -> 266,209 FPS)", ev_s: "Every new throughput envelope plus flat/negative runs plotted on live TPU v5p (inspired by wenhaochai/rtx-h100).",
        gs_t: "LOG-SCALE PIVOT · 3D GAUSSIAN SPLATTING", gs_h: "3DGS Trajectory: Negative Pallas (-1.81x) -> 142k FPS / 300M", gs_s: "Replacing padded Pallas DMA writes with chunked VMEM EWA unlocks 142,008 FPS & 300M splats in 4c HBM.",
        vm_t: "HARDWARE BOUNDARY · LIVE TPU SWEEP", vm_h: "64 MiB VMEM Spill Cliff at tile_size = 512 (-24%)", vm_s: "Measured E2E FPS as ray tile size sweeps 64 -> 128 -> 256 -> 512 on 1 TPU v5p chip (B=1024).",
        mx_t: "MXU QUANTIZATION · ACCURACY VS FPS", mx_h: "FP32 vs BF16 vs INT8 W8A8 MXU Precision Pareto", mx_s: "BF16 doubles throughput (2.11x @ 4K: 78.6 vs 37.2 FPS) with exact 20.276 dB PSNR parity; INT8 QAT costs only -0.047 dB.",
        orb_t: "GENERALIZATION CHECK · CYCLES EXR", orb_h: "Hero Camera vs. Held-Out Orbit Camera PSNR", orb_s: "Reporting both hero-view and full multi-angle held-out orbit parity across 7 Blender Studio scenes.",
        dp_t: "PIXEL-TO-TORQUE · ZERO STATE LABELS", dp_h: "End-to-End jax.grad(mjx.step + PBR) Servoing (240 iters in 8.61 s)", dp_s: "Pure RGB L2 pixel loss drives 7-DOF joint torques from 17.55 dB -> 36.08 dB while cutting EE 3D error from 16.03 cm -> 1.63 cm.",
        pp_t: "ZERO-COPY VISION PPO · 4 TPU CHIPS", pp_h: "Closed-Loop Policy Training in Single-Graph HBM", pp_s: "Cartpole (656.4k steady env-steps/s), PandaPickCube (173.5k/s), and Humanoid-21DOF (221.7k/s) all train end-to-end on TPU pixels.",
        g3_t: "32-VIEW 3DGS GRADIENT CONVERGENCE", g3_h: "Real-World Mip-NeRF 360 garden (25.5 s Wall Clock)", g3_s: "4-chip data-parallel jax.grad hits 28.56 dB in 3.22 s, 30.30 dB in 7.50 s, and 31.31 dB peak @ 23.42 steps/s (749.4 view-steps/s).",
        vl_t: "SIM-TO-VLA, GELSIGHT & AUTONOMOUS RIG", vl_h: "1M VLA Frames (142.6s), ViT-Tiny VLA & 8x1080p+LiDAR Rig", vl_s: "Measured end-to-end multimodal factory data generation, online transformer training, and autonomous perception parity."
      }
    },

    zh: {
      pageTitle: "TPU-RTX — 让 Google TPU 原生跑通光线追踪、路径追踪与 3DGS | Feiyang Liu",
      themeNight: "暗房",
      themeDay: "纸本",
      nav: {
        c1: "01 GPU对比TPU", c2: "02 性能演进", c3: "03 电影与光学", c4: "04 3DGS与神经渲染",
        c5: "05 训练与具身", c6: "06 踩坑与失败案例", c7: "07 账本与索引", c8: "08 极限与边界"
      },
      hero: {
        kicker: "实测于 GOOGLE CLOUD TPU v5p（单芯至 32 芯集群）· ZERO HOST_CALLBACK · ZERO GATHER_OPS",
        by: "作者",
        license: "Apache-2.0 开源",
        title: "<em>TPU-RTX</em>：把 Google TPU 变成原生实时光追引擎。",
        lead: "全栈原生运行于 <strong>Google Cloud TPU v5p</strong>：单张零拷贝 XLA 图直驱 <strong>223 亿</strong>面电影 OpenUSD、<strong>10.6 亿</strong> 3DGS、<strong>171.8 亿</strong>唯一三角面与 <strong>41.1 万 FPS</strong> 具身物理仿真。",
        ctaStart: "先看 GPU vs TPU 架构对比",
        ctaGallery: "数据账本与单场景索引",
        stamp: "全单图全幅专刊 · 零裁剪 · 零多宫格拼图 · 每一帧均来自 TPU v5p 实测",
        t0i: "场景 01 · OPENUSD 日落",
        t0t: "迪士尼 Moana 黄金日落海湾（223 亿面，原生 4K）",
        t1i: "场景 02 · 电影 × MJX",
        t1t: "MJX 人形与四足机器人奔跑在 Moana 沙滩（原生 4K）",
        t2i: "场景 03 · 实景 3DGS",
        t2t: "Mip-NeRF 360 Garden 原生 4K（SH3，零 Gather）"
      },
      kpis: [
        ["4 芯 & 32 芯 MJX+光追", "266k / 411k FPS", "4 芯纯渲染等效 <strong>921k</strong> · <strong>32 芯 E2E 411,023 FPS</strong>"],
        ["迪士尼 Moana OpenUSD", "223 亿三角面", "<strong>4K 5.13 s · 8K 17.24 s</strong> · 日落与水下珊瑚礁 (4 芯)"],
        ["3DGS 速度与集群容量", "95.2 万 FPS / 10.6 亿", "<code>14.2 万</code> (4 芯) / <strong>952,248 FPS</strong> (32 芯 128²) · <strong>10.62 亿</strong>高斯"],
        ["原生 8K 实景与 PT", "7680×4320", "<code>garden 29.19 MPix/s</code> (4 芯) · 8K 路径追踪 <strong>1,650 spm</strong> (32 芯)"],
        ["DLSS 3.5 与 4K 重打光", "58.9 / 78.6 FPS", "DLSS <strong>3.91×</strong> · 零光追重打光 <strong>78.6 FPS</strong> BF16 (4 芯)"],
        ["Blender Cycles 对齐度", "37.65 / 36.98 dB", "<code>BMW27</code> <strong>0.9813</strong> SSIM · <code>Monster</code> <strong>0.121 mm</strong> (4 芯)"],
        ["布料 Refit & 32 芯唯一面", "973 FPS / 171.8 亿", "<code>973.3 FPS</code> 布料 BVH · <strong>171.8 亿</strong>唯一三角面 (32 芯, 1.18 TB)"],
        ["可微物理 + 32 芯 LiDAR", "1.63 cm / 639 亿/s", "<code>jax.grad</code> <strong>36.08 dB</strong> (单芯) · <strong>63.88B</strong> LiDAR 射线/秒 (32 芯)"]
      ],
      c1: {
        num: "01 / 08 · GPU VS TPU",
        rail: "第一性原理对比：为什么传统 SIMT 光追 GPU 与谷歌 TPU 脉动阵列底层逻辑完全不同，各自优势与短板在哪。",
        title: "为什么用 <em>TPU</em>？GPU 与 TPU 底层架构全景对比。",
        lead: "传统 GPU（32 线程 SIMT Warp + 硬件 RT Core + 硬件 L1/L2 缓存）与 Google Cloud TPU v5p（459 TFLOP/s BF16 <strong>128×128 脉动阵列 MXU</strong> + <strong>64 MiB 软件显控 VMEM</strong> + <strong>95.74 GiB 大显存 HBM</strong>）处于硬件设计的两端。照搬 CUDA 思路写 TPU 会瞬间卡死在 HBM 延迟上；但围绕 64 叉 Wide-BVH 与 VMEM 重构后，TPU 在大显存容量、物理-渲染-AI 零拷贝融合上展现出惊人的爆发力。",
        gvtTag: "硬件架构与执行流程图对比",
        gvtTitle: "传统 SIMT RTX GPU vs. Google TPU v5p 脉动阵列光追流水线对比",
        gvtSub: "清晰拆解两者的核心算力单元、缓存层次、BVH 遍历路径与后续 AI/物理仿真耦合方式。",
        gvtRows: [
          ["1. 动态分支与硬件 RT Core", "<span class='badge fail'>劣势</span> 无专用硬件 RT Core，128 通道 VPU 要求规则无分支掩码。", "<span class='badge ok'>优势</span> 拥有硬件 RT Core 与 32 线程 SIMT Warp，天然擅长随机多反弹光线分歧。"],
          ["2. 随机内存 Gather / 原子操作", "<span class='badge fail'>劣势</span> 脊点高达 166 FLOP/B，一旦出现不规则 HBM Gather 或 Tile 原子写，性能会断崖下跌。", "<span class='badge ok'>优势</span> 硬件 L1/L2 缓存与高速片上原子写专门为随机指针跳转和细粒度 Scatter 设计。"],
          ["3. 显存容量 (HBM) 与片上 VMEM", "<span class='badge ok'>优势</span> 单芯 <strong>95.74 GiB HBM（单机 383 GB）</strong> + 64 MiB 显式 VMEM，轻松装下 <strong>3 亿高斯</strong>或 <strong>223 亿面</strong> <em>Moana</em>，无需 PCIe 换页。", "<span class='badge fail'>劣势</span> 单卡显存通常 24–80 GB，大批量具身 RL 或超大规模 3DGS 常碰显存天花板。"],
          ["4. 物理 + 渲染 + RL/VLA 零拷贝融合", "<span class='badge ok'>优势</span> MJX 物理、光追、GelSight、LiDAR、PPO 与 ViT-Tiny 同驻<strong>单张 XLA 计算图</strong>（<code>266k FPS</code>，VLA 仅 <code>-1.84%</code> 开销）。", "<span class='badge fail'>劣势</span> 多跨 Warp / OptiX / Isaac / PyTorch 工具链桥接，常伴随中间缓冲拷出。"],
          ["5. MXU 神经降噪、超分与重打光", "<span class='badge ok'>优势</span> <code>459 TFLOP/s</code> BF16 MXU 直连 HBM G-Buffer：<strong>87.8 FPS</strong> 4K 降噪、<strong>58.9 FPS</strong> 4K DLSS、<strong>78.6 FPS</strong> 重打光。", "<span class='badge ok'>优势</span> Tensor Core + 成熟的官方 DLSS 3.5 与 OptiX OIDN 开箱即用。"],
          ["6. 首次编译延迟 vs 稳态吞吐", "<span class='badge fail'>劣势</span> XLA AOT 编译需 <strong>4–130 秒</strong>冷启动编译，不适合频改拓扑的实时编辑器。", "<span class='badge ok'>优势</span> 亚秒级响应，交互式场景编辑与实时游戏预览体验极佳。"],
          ["7. 生态工具链成熟度", "<span class='badge fail'>劣势</span> 适合在训练/仿真前先编译 <code>.usd</code> / <code>.blend</code> / MJCF / <code>.ply</code> 资产。", "<span class='badge ok'>优势</span> 拥有无可替代的 Omniverse、Blender 实时视口与虚幻引擎生态。"]
        ],
        pillars: [
          ["VMEM 宽树遍历", "1. 三级 64 叉 Wide-BVH", "把 64 个子节点包围盒紧致打包成 (8, 128) 向量块，使 Pallas Mosaic 在单条 VPU 指令内完成 64 路包围盒测试，并配合 pltpu.async_copy 异步流水流式读取叶节点，平稳扩展至 223 亿面 Moana 与 21.5 亿唯一三角面。"],
          ["片上核融合与零 Gather", "2. In-VMEM 核融合与分块 3DGS", "把光线生成、求交、GGX/AO/软阴影与材质去重直接融进 VMEM 寄存器，彻底消除 G-Buffer 往返（92.1 万 FPS 等效）；分块 VMEM take_along_axis 将 3DGS 推至 14.2 万 FPS 与 3 亿高斯。"],
          ["MXU 与端到端可微", "3. 原生 MXU 神经加速与 jax.grad 物理", "像素全程不离 TPU HBM，让 BF16 MXU 运行 87.8 FPS 4K 去噪、58.9 FPS 4K TPU-DLSS 3.5、78.6 FPS 零光追重打光，以及直接穿透 mjx.step 的像素到力矩反向传播。"]
        ],
        diagCap: "全模块在 XLA HLO 层通过硬验证：host_callbacks = 0，send_recv_ops = 0，gather_ops = 0。",
        sdkBar: "极速接入 SDK · tpu_mjx_render.adapters 下内置 8 大生态适配器"
      },
      c2: {
        num: "02 / 08 · 性能演进与泛化",
        rail: "按时间顺序记录每一步优化包络（包含负优化与持平试验）、3DGS 反转、64 MiB VMEM 溢出悬崖、MXU 帕累托与环绕泛化检测。",
        title: "TPU 上 <em>587×</em> 演进：优化轨迹与视角泛化检验。",
        lead: "借鉴 <code>wenhaochai/rtx-h100</code> 的工程透明标准：完整呈现按时间演进的步骤包络曲线（明确标出负优化和持平实验）、<code>tile_size = 512</code> 溢出悬崖（<code>-24%</code>）、BF16/INT8 MXU 帕累托，以及 7 大 Blender 场景的<strong>主视角 vs 环绕泛化</strong> PSNR。"
      },
      c3: {
        num: "03 / 08 · 电影、布料与物理光学",
        rail: "两组全宽左右拉杆对比（6 大 Blender Cycles 真值 + 8 组物理光学与布料）与 5 张完整单场景 4K/8K Moana 与焦散大图。",
        title: "223 亿面 <em>Moana</em>、Cycles 对照、布料与物理光学。",
        lead: "全站彻底去除多宫格拼图：先在下方两组<strong>可交互左右拉杆实验室</strong>中逐像素对照 6 大 Blender Cycles 官方真值与 8 组物理光学/可变形布料变化，再欣赏 5 张全宽完整无裁剪的 Moana 4K/8K 与焦散单场景大图。"
      },
      c4: {
        num: "04 / 08 · 3DGS 与 MXU 神经渲染",
        rail: "原生 4K 实景 3DGS 单图切换大舞台（5 景）+ 4DGS 物理推挤 / TPU-DLSS 3.5 / 4K 降噪 / 重打光 5 组左右拉杆。",
        title: "TPU 原生 <em>3DGS / 4DGS</em>、<code>TPU-DLSS 3.5</code> 与 4K 重打光。",
        lead: "在下方全宽大舞台直接切换 5 个原生 4K 真实重建场景（点击画面开启 1:1 像素放大镜），并拖动 5 组神经滑块对比 <code>4DGS × MJX</code> 推挤、<strong>TPU-DLSS 3.5</strong>（4K <code>58.9 FPS</code>）、<strong>4K 降噪</strong>（<code>87.8 FPS</code>）与<strong>零光追重打光</strong>（<code>78.6 FPS</code>）。"
      },
      c5: {
        num: "05 / 08 · 训练曲线与具身智能",
        rail: "4 组真实 TPU 训练收敛曲线与 MuJoCo Menagerie 全宽单场景 PBR 机器人大舞台（G1、Go2、Panda、Aero Hand）。",
        title: "TPU 显存内闭环训练：<em>可微物理</em>、PPO 与具身机器人。",
        lead: "下方 4 张全宽矢量图直接抽取自 <code>out/metrics/*.json</code> 真机日志，并附上可逐台切换、全宽无裁剪的 <strong>MuJoCo Menagerie</strong> PBR 机器人大舞台（宇树 G1、Go2、Franka Panda、Aero Hand）。"
      },
      c6: {
        num: "06 / 08 · 工程复盘与失败案例",
        rail: "直面真实踩坑：隐藏的 151 MB Gather、反输给 XLA 的初版 Pallas、3 连败才过门的神经着色器，以及被 QA 拦截的真实“翻车图”。",
        title: "TPU 踩坑实录：<em>负优化复盘</em> 与 QA 淘汰样本。",
        lead: "只展示完美成片往往掩盖系统落地的真实难度。这一章完整公开 4 个核心工程踩坑/负优化记录，以及 2 张真实被我们 <strong>Visual QA 门禁淘汰</strong>的反面样本（全宽单张呈现、绝不美化）。",
        pmCards: [
          ["翻车 -> 破局", "E10 / E11 / E12", "1. 以为求交慢，结果 90% 耗时卡在 151 MB 棋盘格 Gather", "早期 Amdahl 剖面揪出惊人真相：286.5 ms 帧耗时里 >90% 根本不在 BVH，而在一块棋盘地面触发的动态 HBM gather geom_xmat[b_idx, safe_gid]。", "改为静态迭代后着色暴降 4.4x（286.5 -> 64.9 ms），单芯直冲 12,843 FPS、4 芯 50,131 FPS（gather_ops = 0）。"],
          ["负结论反演", "E14 vs Wave 3", "2. 手写 Pallas 3DGS 反输给纯 XLA 1.81 倍", "初版 MXU 对数空间 Pallas 核（gs3d_pallas）仅跑 658.6 FPS，大幅落后纯 XLA jax_tile（1,193.2 FPS）——原因是 (BT, 64, 128) 对齐填充让写出膨胀 16 倍且触发 65,536 次碎 DMA。", "先如实保留负结论与 jax_tile 基线，随后重做 VMEM 分块 EWA 排序（gs3d_v3），一举实现 119 倍逆袭（142,008 FPS，4 芯装 3 亿高斯）。"],
          ["3 连败 -> 全过门", "E4 / E5 / E6a -> E6b", "3. Stage-2 神经着色器前 3 轮消融全败", "M1 像素级 MLP 无感受野导致软阴影发虚（27.80 dB）；窄 W32 U-Net 夹爪遮挡错位 10%；Loss 内加 jnp.clip 直接在 Step-1 杀死梯度（卡死 15.20 dB）。", "加入 reflect(V, N) + Fresnel N·V、双 1x1 BRDF 跳连、去 clip 全局梯度裁剪并补强紧握样本后，38.35 dB / 0.9889 SSIM 8/8 项门禁全过。"],
          ["多机踩坑 -> 已实测打通", "32 芯整集群 395.1 s", "4. 跨节点 MJX 非本地数组与多行 SSH 清理残留", "8 机（32 芯）分布式启动时，mjx.put_model(m, impl='jax') 默认 device=None 会生成跨机数组并触发 IFRT 崩溃；早期多行 PID 换行清理曾锁住 /dev/vfio/3。", "已修补 device=local_devs[0] 与单行 tr 清理并清回 8/8 IDLE；随后在获批的 2 小时 8 机独占窗口内仅用 395.07 s 一次跑通全部 32 芯（3.064 TB HBM）压测：41.1 万 FPS E2E、95.2 万 FPS 3DGS、10.62 亿高斯、638.8 亿 LiDAR 射线/秒、171.8 亿唯一三角面与 8K 1,650 spm，并在 15 分钟内清回 8/8 IDLE 交还团队。"]
        ],
        ablationRows: [
          ["Phase-0 锁定硬阈值", ">= 30.0 dB", ">= 28.0 dB", "<= 0.010", "<span class='badge cyan'>必须 8/8 全通过</span>"],
          ["Stage 1 解析直接光基线", "13.83 dB", "—", "—", "仅硬阴影，无软阴影与间接光"],
          ["E4: M1_pixel_128 (仅 1x1 MLP)", "27.80 dB", "23.52 dB", "0.0500 (7/140)", "<span class='badge fail'>FAIL</span> 缺少空间感受野"],
          ["E5: M2_spatial_32 (窄通道 W32)", "35.58 dB", "27.79 dB", "0.1000 (14/140)", "<span class='badge fail'>FAIL</span> 接触区域边缘错位"],
          ["E6a-3: M2_spatial_128 + jnp.clip", "15.20 dB", "—", "—", "<span class='badge fail'>FAIL</span> Step-1 梯度饱和归零"],
          ["E6b: M2_spatial_128 (24-ch BRDF 终版)", "<strong>38.35 dB</strong>", "<strong>30.31 dB</strong>", "<strong>0.00714 (1/140)</strong>", "<span class='badge ok'>PASS (8/8)</span> SSIM 0.9889, 边缘 0.157 px"]
        ],
        rejected: [
          {
            id: "rejected_openusd_clay",
            img: "./assets/postmortem/rejected_openusd_clay.webp",
            banner: "QA 已拦截淘汰 · 早期未绑定贴图的“白模泥塑感”",
            title: "淘汰样本 A：早期 OpenUSD 几何管线直出（v3_w2_openusd_4k）",
            desc: "问题诊断：在完整接入迪士尼 Ptex/材质节点编译之前，远景植被与水面反射退化成平淡白模。已由第 03 章的黄金日落、珊瑚礁焦散与 Moana×MJX 4K 成片全面替代。"
          },
          {
            id: "rejected_garden_seams",
            img: "./assets/postmortem/rejected_garden_seams.webp",
            banner: "QA 已拦截淘汰 · 明显 16x16 TILE 拼缝与黑底裁切",
            title: "淘汰样本 B：早期 32 视角 3DGS 训练可视化（v3_w3_garden_32v_grad）",
            desc: "问题诊断：早期 tile 局部累积边界直接截断了跨 16x16 tile 的高斯尾迹，导致画面出现肉眼可见的方格拼缝与黑底裁切。成片已淘汰并公开展示反面样本，对应 31.31 dB 数值训练日志已单独验证真实。"
          }
        ]
      },
      c7: {
        num: "07 / 08 · 遥测、账本与索引",
        rail: "切换五大子系统实测条形图、零溢出 21 项可溯源 JSON 账本，以及全站 25 张独立大图快速直达目录（无小图缩略堆积）。",
        title: "性能控制台、<em>21 份溯源 JSON</em> 与单场景索引。",
        lead: "每一项核心数据都能直接点开 GitHub <code>out/metrics/*.json</code> 核对（精简 <code>JSON ↗</code> 链接保证表格零横向溢出）；下方目录提供全站 25 张无裁剪单场景的一键灯箱直达，不再重复堆砌小缩略图。"
      },
      c8: {
        num: "08 / 08 · 32 芯集群扩展与署名",
        rail: "32 芯 8 机（3.064 TB HBM）实测对比 4 芯基线、坦诚边界、引自 wenhaochai/rtx-h100 的硬件对照、Feiyang Liu 署名与 BibTeX。",
        title: "<em>32 芯 TPU 集群</em>（3.06 TB 显存）、边界与 <code>rtx-h100</code>。",
        lead: "汇总 8 机 32 芯 Pod Capstone（<code>3.064 TB</code> HBM，全程 <code>395.07 s</code> 实测收口）与单机 4 芯基线（<code>383 GB</code> HBM）扩展对照、引用 <code>wenhaochai/rtx-h100</code> 的坦诚对比，以及 Feiyang Liu BibTeX 引用卡。",
        honest: "全页所有指标均来自真机 TPU v5p 实测（单芯、单机 4 芯或 8 机 32 芯分布式集群均已明确标注，零注水外推）。同时坦诚写明 GPU 依然领先的领域：亚秒级互动视口响应、硬件 RT Core 应对随机多反弹分支，以及单卡 Cycles 极速渲染（Chai et al., 2026）。",
        authorBio: "聚焦编译器原生图形渲染、可微物理仿真与 TPU/AI 加速器高吞吐具身智能系统的研究者与工程师。",
        ack: "第三方资产与参考致谢：Walt Disney Animation Studios Moana Island Scene v2.1（非商业研究许可）；Blender Open Data / Blender Studio 场景（BMW27、Barcelona Pavilion、Barbershop、Classroom、Junkshop、Lone Monk、Monster Under the Bed、Fishy Cat — CC0 / CC-BY）；Mip-NeRF 360 真实摄影测量数据集；Google DeepMind mujoco_menagerie（Apache-2.0 / BSD）；Chai 等（2026）wenhaochai/rtx-h100 H100/RTX PRO 6000 Cycles 基准数据。"
      },
      podStats: [
        ["基准 1 & 2 · 具身物理 + 3DGS (4 芯 -> 32 芯)", "411,023 FPS E2E  |  952,248 FPS 3DGS", "从 4 芯 5.15 万 FPS 接近线性扩展 8x 至 32 芯 41.10 万 FPS（B=131k）；128² 3DGS 达 95.22 万 FPS，外加 ICI 分布式 10.62 亿高斯 @ 1080p（3.41 s）。"],
        ["基准 3 · 128x2048 批量 360° 激光雷达 (32 芯)", "243,676 env-FPS  |  每秒 638.8 亿条射线", "8 机 32 芯上 B=16,384 并发 128x2048 线 LiDAR 仅 67.2 ms（B=32,768 达 229,060 env-FPS / 600.5 亿射线/秒，gather_ops=0）。"],
        ["基准 4 · 几何分片唯一三角面上限 (32 芯)", "171.8 亿唯一非实例化三角面 (1.18 TB HBM)", "ICI jax.lax.pmin 相对 4 芯 21.5 亿实现整整 8x 扩展：171.8 亿唯一面 4K 4.46 s / 8K 8.53 s（85.9 亿面 4K 2.43 s / 8K 4.70 s）。"],
        ["基准 5 · 空间分片原生 8K 3-Bounce 路径追踪 (32 芯)", "7680x4320 达 1,650.2 spm (7.54x 扩展)", "3,318 万像素无偏 3 次反弹蒙特卡洛全局光（5.475 GRays/s）：4 spp 仅 145.4 ms（6.88 FPS），1 spp 达 37.5 ms（26.69 FPS，零回调）。"]
      ],
      compRows: [
        ["具身视觉 RL、VLA 与触觉", "<strong>单张 XLA 零拷贝图</strong>：端到端 <code>266k FPS</code>、142.6s 生成百万帧、GelSight <code>270.6k/s</code>、Sim-to-VLA 仅 <code>-1.84%</code> 开销", "静态离线渲染器（无交互式 RL 闭环）", "成熟的 CUDA Warp / Isaac 生态与 RTX/H100 工具链"],
        ["Real-to-Sim 3DGS / 4DGS + 物理", "<strong>零 Gather 原生 TPU</strong>：<code>142k FPS</code>（<code>64²</code>）、4 芯 <code>3 亿</code>高斯 4K、<code>8K garden 29.2 MPix/s</code>、实时 <code>4DGS x MJX</code> 推挤", "仅支持 Cycles Mesh + NRC", "需要独立 CUDA gsplat 与 USD/PhysX 跨端桥接"],
        ["端到端可微物理 + 光追渲染", "<strong>36.08 dB / 末端误差 1.63 cm</strong>：<code>jax.value_and_grad(mjx.step + PBR)</code> 实测 <code>27.9 grad-steps/s</code>", "不可微离线路径追踪器", "需单独组合 Warp 可微物理与 Nvdiffrast"],
        ["电影 OpenUSD 与 Blender 路径追踪", "<strong>超大场景编译上限</strong>：迪士尼 <em>Moana</em>（<code>223 亿</code>面仅 4.65 GiB）、<code>BMW27 37.65 dB</code>、<code>Monster 36.98 dB</code>", "<strong>绝对速度标杆 (wenhaochai/rtx-h100)</strong>：单卡 H100 达 <code>55,148 spm</code>（7.8× Blender），RTX PRO 6000 OptiX 达 <code>78,297 spm</code>（NRC 39.9 img/s）", "完整 Omniverse / OptiX 实时互动视口"],
        ["AI 重打光、降噪与超分 DLSS", "<strong>原生 MXU BF16/INT8</strong>：4K 降噪 <code>87.8 FPS</code>、4K DLSS <code>58.9 FPS</code>、零光追重打光 4K <code>78.6 FPS</code> / 1080p <code>307 FPS</code>", "NRC-scatter <code>40.4 img/s</code>（H100 32-spp）", "专用 Tensor Core 硬件 DLSS 3.5 与光线重建"]
      ],
      ui: {
        copy: "复制", copied: "已复制", zoom4k: "检视原图 ↗", loupe: "1:1 像素放大镜 (L)", raw: "打开完整原图 ↗",
        footerL: "TPU-RTX · Google Cloud TPU v5p (JAX + Pallas Mosaic · 零 host_callbacks · 零 HBM Gather)",
        searchPlaceholder: "搜索实验名称、场景或 JSON 文件名...",
        b_mjx: "MJX 与具身仿真 (w0/w1/w6)", b_gs3d: "3DGS / 4DGS / 3 亿高斯 (w3)", b_cinema: "Moana / 布料 / Blender (w2/w5/w7)", b_dlss: "TPU-DLSS 3.5 与重打光 (w4)", b_fx: "路径追踪 / ReSTIR / 光学 (w7/w6)",
        g_all: "全部单场景 (25)", g_film: "Moana 与 Blender Studio (10)", g_optics: "物理光学与焦散 (4)", g_gs3d: "实景 3DGS / Bistro (5)", g_robotics: "Menagerie PBR 机器人 (4)", g_rej: "QA 淘汰案例 (2)",
        th1: "任务类型", th2: "TPU-RTX (Google TPU v5p)", th3: "GPU Cycles (wenhaochai/rtx-h100)", th4: "NVIDIA Isaac Sim / OptiX",
        gvt1: "对比维度", gvt2: "Google Cloud TPU v5p (TPU-RTX)", gvt3: "NVIDIA GPU (RTX / H100 / Isaac)",
        ab1: "Stage-2 神经着色器消融阶段", ab2: "PSNR (dB)", ab3: "夹爪接触区 PSNR", ab4: "遮挡错位率", ab5: "门禁结果与归因",
        ld_all: "全部 (21)", ld_rob: "具身与仿真", ld_film: "电影与布料", ld_gs: "3DGS/4DGS", ld_nn: "神经/光学",
        lt1: "实验项目", lt2: "配置与规模", lt3: "实测吞吐", lt4: "精度与对齐状态", lt5: "JSON",
        ev_t: "时间顺序步骤包络图 · 公开负优化实验", ev_h: "587x 光追管线演进曲线（453 -> 266,209 FPS）", ev_s: "按时间标注每次新高及平缓/负优化实验（参考 wenhaochai/rtx-h100 图表范式）。",
        gs_t: "对数尺度反转 · 3D 高斯溅射", gs_h: "3DGS 演进曲线：从 Pallas 负结论（-1.81x）逆袭至 14.2 万 FPS / 3 亿高斯", gs_s: "放弃膨胀写出的初版 Pallas 后重构 VMEM 分块 EWA，实现 142,008 FPS 与 3 亿高斯同驻显存。",
        vm_t: "硬件物理边界 · 真机扫描", vm_h: "64 MiB VMEM 溢出悬崖（tile_size = 512 骤降 -24%）", vm_s: "单芯 TPU v5p（B=1024）下随 tile_size 从 64 扫至 512 的 E2E FPS 实测。",
        mx_t: "MXU 量化权衡 · 精度 vs 帧率", mx_h: "FP32 vs BF16 vs INT8 W8A8 MXU 精度与吞吐帕累托", mx_s: "BF16 在零 PSNR 掉点（20.276 dB）下将原生 4K 帧率翻倍至 2.11x（78.6 vs 37.2 FPS）；INT8 QAT 仅降 0.047 dB。",
        orb_t: "视角泛化检验 · CYCLES EXR", orb_h: "主视角 vs 多角度留出环绕轨迹 PSNR 对齐", orb_s: "同时公开单一主视角与多机位留出环绕平均 PSNR，防止过拟合单一角度。",
        dp_t: "纯像素反传力矩 · 零关节状态监督", dp_h: "端到端 jax.grad(mjx.step + 光追) 视觉伺服（8.61 s 收敛）", dp_s: "仅凭 RGB L2 误差驱动 7 自由度力矩，画质 17.55 -> 36.08 dB、末端 3D 误差从 16.03 -> 1.63 cm。",
        pp_t: "零拷贝闭环 Vision PPO · 4 芯实测", pp_h: "单图 HBM 强化学习训练收敛曲线", pp_s: "Cartpole（稳态 656.4k/s）、Panda（173.5k/s）、Humanoid-21DOF（221.7k/s）全片上实时渲染+更新。",
        g3_t: "32 视角 3DGS 快速梯度训练", g3_h: "真实 Mip-NeRF 360 garden 25.5 s 训练收敛曲线", g3_s: "4 芯数据并行 jax.grad：3.22 s 破 28.56 dB、7.50 s 破 30.30 dB、17.11 s 达 31.31 dB 峰值（23.42 steps/s）。",
        vl_t: "SIM-TO-VLA、视触觉与自动驾驶传感器组", vl_h: "142.6s 百万 VLA 帧、ViT-Tiny 训练 & 8x1080p+128 线雷达", vl_s: "百万帧数据工厂、融合 VLA（-1.84% 开销）、4 路 GelSight（270.6k/s）及环视+双激光雷达实时真测。"
      }
    }
  }
};
