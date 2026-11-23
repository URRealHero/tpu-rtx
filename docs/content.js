/* ==========================================================================
   TPU-RTX Bilingual Content Dictionary (English / 中文)
   6-Chapter Monograph Architecture:
   - 26 Clean Standalone Single-Scene Plates (Zero Defects, Zero Collages)
   - 3 Interactive Split Slider Labs (19 Aligned Left/Right Presets)
   - 3 Interactive Full-Width Single-Scene Stages (Hero, 4K 3DGS, Robots PBR)
   - 11 Full-Width Vector SVGs, 4 Post-Mortems, 21-Row JSON Ledger
   Author: Feiyang Liu
   ========================================================================== */

window.TPU_RTX_DATA = {
  plates: [
    // --- Moana & Film / Optics Singles (10 Film) ---
    {
      id: "moana_daylight_4k",
      cat: "film",
      web: "./assets/singles/moana_daylight_4k.webp",
      png: "./assets/singles/moana_daylight_4k.webp",
      badges: {
        en: [["hot", "Moana Daylight 4K"], ["", "22.30B Instanced Tris"], ["ok", "4.65 GiB HBM · 5.13 s"]],
        zh: [["hot", "晴天 Moana 海湾 4K"], ["", "223 亿实例化三角面"], ["ok", "4.65 GiB 显存 · 5.13 s"]]
      },
      title: {
        en: "Disney Moana OpenUSD v2.1: Motunui Turquoise Lagoon Daylight (4K)",
        zh: "迪士尼 Moana OpenUSD v2.1：Motunui 晴天绿松石珊瑚泻湖（原生 4K）"
      },
      desc: {
        en: "Full daylight wide view of Disney's 22.30B-tri OpenUSD island showing transparent turquoise lagoon water, submerged coral heads, palm canopy, and volcanic ridges compiled into 4.65 GiB TPU HBM (5.13 s @ 4K).",
        zh: "晴天广角视角直出迪士尼 223 亿面 OpenUSD 岛屿：清澈绿松石海湾水体、水下礁石、热带椰林与火山山脊全量装入 4.65 GiB TPU HBM（原生 4K 5.13 s）。"
      }
    },
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
        en: "Disney Moana OpenUSD @ Native 8K Tropical Palm Canopy & Ridge",
        zh: "迪士尼 Moana OpenUSD 原生 8K（7680x4320）热带棕榈冠层与山脊特写"
      },
      desc: {
        en: "Full 33.18M-pixel 8K raytrace through dense instanced tropical palms and mountain foliage directly via OpenUSD PointInstancers across 4 TPU v5p chips in 17.24 s (3.85 MRays/s, gather_ops = 0).",
        zh: "4 颗 TPU v5p 芯片协同直出完整 7680x4320 像素 8K 高密热带棕榈冠层与山脊树叶，直接遍历 OpenUSD PointInstancer（17.24 s，gather_ops = 0）。"
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
    // --- Physical Optics Singles (3 Optics) ---
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
    // --- Real-World 3DGS Singles (6 Real-World 3DGS) ---
    {
      id: "robowild_panda_4k",
      cat: "gs3d",
      web: "./assets/singles/robowild_panda_4k.webp",
      png: "./assets/singles/robowild_panda_4k.webp",
      badges: {
        en: [["hot", "RoboWild 4K Composite"], ["", "1.80M Kitchen + Panda"], ["ok", "Zero HUD · Exact Z"]],
        zh: [["hot", "RoboWild 4K 实景融合"], ["", "180 万厨房高斯 + Panda"], ["ok", "无遮挡原图 · 精准深度"]]
      },
      title: {
        en: "RoboWild 4K: Live MJX Franka Panda inside Mip-NeRF 360 Kitchen + LEGO Dozer",
        zh: "RoboWild 原生 4K：MJX Franka Panda 直接部署进 Mip-NeRF 360 厨房乐高实景"
      },
      desc: {
        en: "Borderless 4K composite of a live MuJoCo MJX Franka Panda casting analytical shadows onto real tabletop placemats alongside the 1.80M-splat LEGO Dozer capture.",
        zh: "零 HUD 遮挡原生 4K 混合成片：实时 MJX Franka 机械臂在真实 Mip-NeRF 360 厨房桌布上投下精准投影并靠近 180 万高斯乐高推土机。"
      }
    },
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
    // --- MuJoCo Menagerie & RoboTwin PBR Robot Singles (5 Robotics) ---
    {
      id: "robotwin_bimanual",
      cat: "robotics",
      web: "./assets/singles/robotwin_bimanual.webp",
      png: "./assets/singles/robotwin_bimanual.webp",
      badges: {
        en: [["hot", "RoboTwin 16-DOF"], ["", "123.9k cam-FPS (DR)"], ["ok", "72 MJX Geoms + GGX"]],
        zh: [["hot", "RoboTwin 双臂桌面"], ["", "12.39 万 cam-FPS (DR)"], ["ok", "72 刚体 + GGX 软阴影"]]
      },
      title: {
        en: "RoboTwin 16-DOF Bimanual Tabletop — Isaac Replicator DR + Wide-BVH-64",
        zh: "RoboTwin 16 自由度双臂桌面操控 — Isaac Replicator DR + Wide-BVH-64"
      },
      desc: {
        en: "Dual 7-DOF bimanual arms, overhead third-person cam, colored sorting bins and tabletop props executing at 123,858 cam-FPS (41,286 env-steps/s) across 4 TPU v5p chips with only +0.32% DR overhead.",
        zh: "双臂协作抓取、分拣料盒与桌面积木场景在 4 芯 TPU v5p 上跑出 123,858 cam-FPS（41,286 env-steps/s），五模态 Replicator DR 仅增加 +0.32% 开销。"
      }
    },
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

  spreadCh3: ["moana_daylight_4k", "moana_sunset", "moana_8k_bay", "prism_dispersion", "cathedral_godrays"],

  videoStageCh3: [
    {
      id: "mipnerf_garden_4k",
      cat: "gs3d",
      mp4: "./assets/videos/mipnerf_garden_4k.mp4",
      poster: "./assets/videos/mipnerf_garden_4k_poster.jpg",
      tabIdx: { en: "4K VIDEO · 5.83M SH3", zh: "4K 实拍重建 · 583 万高斯" },
      tabTitle: { en: "Mip-NeRF 360 Garden (4K Orbit)", zh: "Mip-NeRF 360 Garden 花园 360° (4K)" },
      hudL: { en: "<strong>Mip-NeRF 360 Garden (30 fps Native 4K)</strong> · 5,834,784 Full SH3 Gaussians · 4 TPU v5p Chips", zh: "<strong>Mip-NeRF 360 花园实景（原生 4K 30fps）</strong>· 583 万全 SH3 高斯 · 4 芯 TPU v5p 实时光栅" },
      hudR: { en: "<strong>3840×2160 Native 4K</strong> · Zero HBM Gathers · Zero Baked Text", zh: "<strong>3840×2160 原生 4K</strong> · 零 HBM Gather · 纯净全幅" }
    },
    {
      id: "robowild_4dgs_push_4k",
      cat: "gs3d",
      mp4: "./assets/videos/robowild_4dgs_push_4k.mp4",
      poster: "./assets/videos/robowild_4dgs_push_4k_poster.jpg",
      tabIdx: { en: "4K VIDEO · 4DGS × MJX", zh: "4K 视频 · 4DGS 物理交互" },
      tabTitle: { en: "RoboWild Panda × 4DGS Dozer (4K)", zh: "RoboWild 机械臂推 4DGS 推土机 (4K)" },
      hudL: { en: "<strong>Live RoboWild 4DGS × MJX Contact Push (4K)</strong> · SE(3) Rigid Gaussians + Analytical Panda Shadows", zh: "<strong>实景 4DGS × MJX 机器人触碰推挤（原生 4K）</strong>· 乐高推土机 SE(3) 联动 + 桌面投影" },
      hudR: { en: "<strong>3840×2160 Native 4K</strong> · Physics + 3DGS Single XLA Graph", zh: "<strong>3840×2160 原生 4K</strong> · 物理+光栅单 XLA 计算图" }
    },
    {
      id: "mipnerf_bicycle_4k",
      cat: "gs3d",
      mp4: "./assets/videos/mipnerf_bicycle_4k.mp4",
      poster: "./assets/videos/mipnerf_bicycle_4k_poster.jpg",
      tabIdx: { en: "4K VIDEO · 6.13M SH3", zh: "4K 实景 · 613 万高斯" },
      tabTitle: { en: "Mip-NeRF 360 Bicycle (4K Orbit)", zh: "Mip-NeRF 360 Bicycle 公园自行车 (4K)" },
      hudL: { en: "<strong>Mip-NeRF 360 Bicycle (Native 4K)</strong> · 6,131,954 SH3 Splats · Crisp Rim Spokes &amp; Foliage", zh: "<strong>Mip-NeRF 360 公园自行车（原生 4K）</strong>· 613 万 SH3 高斯 · 草坪树冠与细辐条" },
      hudR: { en: "<strong>3840×2160 Native 4K</strong> · 4-Chip TPU v5p · gather_ops=0", zh: "<strong>3840×2160 原生 4K</strong> · 4 芯 TPU v5p · 零 Gather" }
    },
    {
      id: "mipnerf_bonsai_4k",
      cat: "gs3d",
      mp4: "./assets/videos/mipnerf_bonsai_4k.mp4",
      poster: "./assets/videos/mipnerf_bonsai_4k_poster.jpg",
      tabIdx: { en: "4K VIDEO · 1.24M SH3", zh: "4K 实景 · 室内樱花盆景" },
      tabTitle: { en: "Mip-NeRF 360 Bonsai (4K Orbit)", zh: "Mip-NeRF 360 Bonsai 桌面盆景 (4K)" },
      hudL: { en: "<strong>Mip-NeRF 360 Bonsai (Native 4K)</strong> · 1,244,819 SH3 Splats · Studio Table 360° Fly-Around", zh: "<strong>Mip-NeRF 360 桌面樱花盆景（原生 4K）</strong>· 124 万 SH3 高斯 · 360° 平滑环绕" },
      hudR: { en: "<strong>3840×2160 Native 4K</strong> · 4.20 FPS Live 4K Raster", zh: "<strong>3840×2160 原生 4K</strong> · 4 芯真机实测 4.20 FPS" }
    },
    {
      id: "mipnerf_counter_orbit",
      cat: "gs3d",
      mp4: "./assets/videos/mipnerf_counter_orbit.mp4",
      poster: "./assets/videos/mipnerf_counter_orbit_poster.jpg",
      tabIdx: { en: "1080P · KITCHEN TABLE", zh: "1080p · 厨房料理台" },
      tabTitle: { en: "Mip-NeRF Countertop (1080p Orbit)", zh: "Mip-NeRF 360 Counter 料理台环绕" },
      hudL: { en: "<strong>Mip-NeRF 360 Counter (1920×1080, 120f)</strong> · Specular Jar, Bread &amp; Baking Tray SH3 Orbit", zh: "<strong>Mip-NeRF 360 厨房料理台（1080p，120 帧）</strong>· 玻璃壶、面点与金属烤盘 SH3 视角响应" },
      hudR: { en: "<strong>1920×1080 30fps</strong> · Two-Band Exact EWA Splatting", zh: "<strong>1920×1080 30fps</strong> · 双频精确 EWA 无截断投影" }
    },
    {
      id: "mipnerf_stump_orbit",
      cat: "gs3d",
      mp4: "./assets/videos/mipnerf_stump_orbit.mp4",
      poster: "./assets/videos/mipnerf_stump_orbit_poster.jpg",
      tabIdx: { en: "1080P · 4.96M SH3", zh: "1080p · 496 万树桩" },
      tabTitle: { en: "Mip-NeRF Forest Stump (1080p)", zh: "Mip-NeRF 360 Stump 森林树桩" },
      hudL: { en: "<strong>Mip-NeRF 360 Forest Stump (1080p)</strong> · 4,961,797 SH3 Gaussians · Ivy &amp; Bark Micro-Geometry", zh: "<strong>Mip-NeRF 360 森林老树桩（1080p）</strong>· 496 万全 SH3 高斯 · 常春藤与树皮细节" },
      hudR: { en: "<strong>1920×1080 30fps</strong> · 62.4 s Total TPU Render (150f)", zh: "<strong>1920×1080 30fps</strong> · 4 芯实测 150 帧仅 62.4 s" }
    },
    {
      id: "thinlens_bokeh_dolly_4k",
      cat: "optics",
      mp4: "./assets/videos/thinlens_bokeh_dolly_4k.mp4",
      poster: "./assets/videos/thinlens_bokeh_dolly_4k_poster.jpg",
      tabIdx: { en: "4K OPTICS · f/1.4 DOLLY", zh: "4K 光学 · f/1.4 移轴拉焦" },
      tabTitle: { en: "4K f/1.4 Thin-Lens Parallax + Focus Rack", zh: "4K f/1.4 薄透镜横移视差+变焦" },
      hudL: { en: "<strong>Stochastic Thin-Lens Camera (Native 4K, 32 spp)</strong> · Concentric-Disk Aperture + R2 Cranley-Patterson", zh: "<strong>原生 4K 随机薄透镜光圈（f/1.4，32 spp）</strong>· Shirley-Chiu 同心圆盘采样 + 横移视差" },
      hudR: { en: "<strong>3840×2160 Native 4K</strong> · Single-Compile Mosaic Pallas", zh: "<strong>3840×2160 原生 4K</strong> · 单次编译零主控回调" }
    },
    {
      id: "thinlens_bokeh_pull_4k",
      cat: "optics",
      mp4: "./assets/videos/thinlens_bokeh_pull_4k.mp4",
      poster: "./assets/videos/thinlens_bokeh_pull_4k_poster.jpg",
      tabIdx: { en: "4K OPTICS · FOCUS RACK", zh: "4K 光学 · f/1.4 景深虚化" },
      tabTitle: { en: "4K f/1.4 Near→Far→Near Focus Pull", zh: "4K f/1.4 前景↔背景光斑平滑拉焦" },
      hudL: { en: "<strong>55mm f/1.4 Focus Rack (0.78 m ↔ 2.25 m)</strong> · Foreground Gold Sphere to Luminous Bokeh Orbs", zh: "<strong>55mm f/1.4 景深呼吸拉焦（0.78 m ↔ 2.25 m）</strong>· 前景金属球至背景霓虹焦外圆斑" },
      hudR: { en: "<strong>3840×2160 Native 4K</strong> · 32 spp HDR ACES Filmic", zh: "<strong>3840×2160 原生 4K</strong> · 32 spp HDR ACES 映射" }
    },
    {
      id: "cloth_bvh_refit",
      cat: "optics",
      mp4: "./assets/videos/cloth_bvh_refit.mp4",
      poster: "./assets/videos/cloth_bvh_refit_poster.jpg",
      tabIdx: { en: "LIVE OPTIX · BVH REFIT", zh: "动态 Optix · 布料 BVH 重拟合" },
      tabTitle: { en: "PBD Silk Cloth Live Wide-BVH Refit", zh: "PBD 丝绸布料碰撞与 BVH 重拟合" },
      hudL: { en: "<strong>Real-Time Deformable Silk Cloth (32,768 Tris)</strong> · Bottom-Up L0→L1→L2 Wide-BVH-64 Refit Inside XLA", zh: "<strong>实时可变形 PBD 丝绸布料（32,768 面）</strong>· XLA 图内从叶至根自底向上重拟合 BVH" },
      hudR: { en: "<strong>0.0 m Depth Error</strong> vs CPU Rebuild · Soft Shadow + Sheen", zh: "对比 CPU 重建 <strong>0.0 m</strong> 误差 · 实时软阴影与丝绸光泽" }
    },
    {
      id: "relight_tod_sweep",
      cat: "relight",
      mp4: "./assets/videos/relight_tod_sweep.mp4",
      poster: "./assets/videos/relight_tod_sweep_poster.jpg",
      tabIdx: { en: "MXU BF16 · ZERO-RT", zh: "BF16 MXU · 零光追昼夜切换" },
      tabTitle: { en: "Classroom Sunlit Time-of-Day Sweep", zh: "教室百叶窗零光追昼夜连续变换" },
      hudL: { en: "<strong>Single Static G-Buffer Interactive Relight</strong> · Morning → Noon → 2600K Sunset → 9500K Twilight", zh: "<strong>单一静态 G-Buffer 零光追实时重打光</strong>· 清晨 → 正午 → 黄金日落 → 蓝调暮光连续滑动" },
      hudR: { en: "<strong>0 Raytrace Ops / Frame</strong> · 307 FPS 1080p / 78.6 FPS 4K", zh: "<strong>逐帧 0 次光追</strong> · 1080p 307 FPS / 4K 78.6 FPS" }
    },
    {
      id: "relight_material_sweep",
      cat: "relight",
      mp4: "./assets/videos/relight_material_sweep.mp4",
      poster: "./assets/videos/relight_material_sweep_poster.jpg",
      tabIdx: { en: "MXU BF16 · BRDF SWEEP", zh: "BF16 MXU · 材质连续转换" },
      tabTitle: { en: "Cornell Workshop Material Sweep", zh: "康奈尔工坊材质哑光→镀铬连贯变化" },
      hudL: { en: "<strong>Zero-Raytrace FiLM BRDF Relighting</strong> · Matte Clay → Satin → Brushed Gold → Mirror Chrome", zh: "<strong>零光追 FiLM 连续材质推理</strong>· 哑光泥塑 → 缎面 → 拉丝金 → 高光镜面铬合金" },
      hudR: { en: "<strong>0 Raytrace Ops / Frame</strong> · Verified vs 4-Bounce MC GT", zh: "<strong>逐帧 0 次光追</strong> · 对齐 4-Bounce 蒙特卡洛真值" }
    }
  ],

  videoStageCh5: [
    {
      id: "robotwin_bimanual",
      mp4: "./assets/videos/robotwin_bimanual.mp4",
      poster: "./assets/videos/robotwin_bimanual_poster.jpg",
      tabIdx: { en: "EMBODIED · ROBOTWIN", zh: "具身仿真 · ROBOTWIN 双臂" },
      tabTitle: { en: "RoboTwin 16-DOF Bimanual Manipulation", zh: "RoboTwin 16 自由度双臂桌面操控" },
      hudL: { en: "<strong>RoboTwin Bimanual Tabletop (1024×1024 AgX)</strong> · Live Articulated Dual-Arm Trajectory + Studio Shadows", zh: "<strong>RoboTwin 双臂桌面操控（1024×1024 AgX）</strong>· 72 刚体实时碰撞影 + 工具台精细光影" },
      hudR: { en: "<strong>123,858 cam-FPS</strong> DR Bench · Zero CPU Callback", zh: "4 芯 3 摄 DR 实测 <strong>123,858 cam-FPS</strong>" }
    },
    {
      id: "diffphys_servoing",
      mp4: "./assets/videos/diffphys_servoing.mp4",
      poster: "./assets/videos/diffphys_servoing_poster.jpg",
      tabIdx: { en: "JAX.GRAD · DIFF-PHYS", zh: "可微物理+渲染 · 视觉伺服" },
      tabTitle: { en: "Pixel-to-Torque jax.grad Servoing (1024²)", zh: "纯像素反传力矩视觉伺服收敛全过程" },
      hudL: { en: "<strong>240-Iter jax.value_and_grad(mjx.step + PBR)</strong> · Torque Solved from RGB Loss Alone (Zero State Labels)", zh: "<strong>240 步 jax.value_and_grad(物理+光追) 优化轨迹</strong>· 零关节状态监督，仅凭图像误差反传力矩" },
      hudR: { en: "<strong>36.08 dB</strong> · <strong>1.63 cm EE Error</strong> in 8.61 s", zh: "8.61 秒收敛至 <strong>36.08 dB</strong> / 末端误差 <strong>1.63 cm</strong>" }
    },
    {
      id: "menagerie_g1_orbit",
      mp4: "./assets/videos/menagerie_g1_orbit.mp4",
      poster: "./assets/videos/menagerie_g1_orbit_poster.jpg",
      tabIdx: { en: "MENAGERIE · UNITREE G1", zh: "MENAGERIE · 宇树 G1 人形" },
      tabTitle: { en: "Unitree G1 Humanoid 360° (1024²)", zh: "宇树 Unitree G1 人形机器人 360°" },
      hudL: { en: "<strong>Official Google DeepMind MuJoCo Menagerie Unitree G1</strong> · 393k Tris · Wide-BVH-64 + Hemisphere AO", zh: "<strong>DeepMind MuJoCo Menagerie 官方宇树 G1</strong>· 39.3 万三角面 · Wide-BVH-64 + 半球 AO" },
      hudR: { en: "<strong>2,546 FPS</strong> @ 64² (4 Chips) · Direct MJCF", zh: "4 芯 64² 实测 <strong>2,546 FPS</strong> · 官方 MJCF 直编" }
    },
    {
      id: "menagerie_go2_orbit",
      mp4: "./assets/videos/menagerie_go2_orbit.mp4",
      poster: "./assets/videos/menagerie_go2_orbit_poster.jpg",
      tabIdx: { en: "MENAGERIE · UNITREE GO2", zh: "MENAGERIE · 宇树 Go2 四足" },
      tabTitle: { en: "Unitree Go2 Quadruped 360° (1024²)", zh: "宇树 Unitree Go2 四足机器狗 360°" },
      hudL: { en: "<strong>Official MuJoCo Menagerie Unitree Go2 (398k Tris)</strong> · Embossed Chassis Mesh + Live Soft Shadows", zh: "<strong>官方 MuJoCo Menagerie 宇树 Go2（39.8 万面）</strong>· 机身细节雕纹 + 实时软阴影" },
      hudR: { en: "<strong>1,863 FPS</strong> @ 64² (4 Chips) · gather_ops=0", zh: "4 芯 64² 实测 <strong>1,863 FPS</strong> · 零 HBM Gather" }
    },
    {
      id: "humanoid_running_orbit",
      mp4: "./assets/videos/humanoid_running_orbit.mp4",
      poster: "./assets/videos/humanoid_running_orbit_poster.jpg",
      tabIdx: { en: "MJX STUDIO · HUMANOID", zh: "MJX 物理 · 人形奔跑步态" },
      tabTitle: { en: "21-DOF Humanoid Running Stride (1024²)", zh: "21 自由度人形机器人奔跑步态 360°" },
      hudL: { en: "<strong>27-DOF Humanoid Studio Running Cycle</strong> · 16-Sample AO + Penumbra Soft Shadow + AgX Grade", zh: "<strong>27 自由度人形奔跑动画</strong>· 16 采样 AO + 半影软阴影 + Blender AgX 电影调色" },
      hudR: { en: "<strong>292,000 env/s</strong> Vision PPO Trained in HBM", zh: "4 芯片上 Vision PPO 达 <strong>29.2 万 env/s</strong>" }
    },
    {
      id: "quadruped_trot_orbit",
      mp4: "./assets/videos/quadruped_trot_orbit.mp4",
      poster: "./assets/videos/quadruped_trot_orbit_poster.jpg",
      tabIdx: { en: "MJX STUDIO · QUADRUPED", zh: "MJX 物理 · 四足对角小跑" },
      tabTitle: { en: "12-DOF Quadruped Diagonal Trot (1024²)", zh: "12 自由度四足机器人小跑步态 360°" },
      hudL: { en: "<strong>Quadruped Diagonal Trot Gait (1024×1024)</strong> · Checkerboard Floor + Dynamic Joint Kinematics", zh: "<strong>四足对角小跑动态步态（1024×1024）</strong>· 棋盘地面高光阴影实时追踪" },
      hudR: { en: "<strong>4-Chip Zero-Copy TPU-RTX</strong> · host_callbacks=0", zh: "<strong>4 芯零拷贝 TPU-RTX</strong> · 零主机回调" }
    }
  ],

  stage3dgs: [
    {
      plateId: "robowild_panda_4k",
      tabIdx: { en: "SCENE 01 · ROBOWILD 4K", zh: "场景 01 · ROBOWILD 4K" },
      tabTitle: { en: "RoboWild Panda × Kitchen (4K)", zh: "RoboWild Panda × 厨房 (4K)" },
      hudL: { en: "<strong>RoboWild Franka Panda × Mip-NeRF Kitchen</strong> · 1.80M SH3 Splats · Zero HUD", zh: "<strong>RoboWild Franka 机械臂 × Mip-NeRF 厨房</strong> · 180 万 SH3 高斯 · 无遮挡原图" },
      hudR: { en: "Native <strong>4K (3840x2160)</strong> · Exact MJX Shadow &amp; Z-Depth · Click to Zoom", zh: "原生 <strong>4K (3840x2160)</strong> · 精准 MJX 阴影与深度融合 · 点击放大" }
    },
    {
      plateId: "garden_4k",
      tabIdx: { en: "SCENE 02 · OUTDOOR", zh: "场景 02 · 户外花园" },
      tabTitle: { en: "garden (5.83M SH3 · 2.61 FPS)", zh: "garden 航拍花园 (583 万高斯)" },
      hudL: { en: "<strong>Mip-NeRF 360 garden</strong> · 5,834,784 SH3 Splats · gather_ops = 0", zh: "<strong>Mip-NeRF 360 garden</strong> · 583 万全 SH3 高斯 · 零 Gather" },
      hudR: { en: "Native <strong>4K @ 2.61 FPS</strong> · <strong>8K @ 29.19 MPix/s</strong> · Click to Zoom", zh: "原生 <strong>4K 2.61 FPS</strong> · <strong>8K 29.19 MPix/s</strong> · 点击放大" }
    },
    {
      plateId: "bicycle_4k",
      tabIdx: { en: "SCENE 03 · FOLIAGE", zh: "场景 03 · 细草坪与辐条" },
      tabTitle: { en: "bicycle (6.13M SH3 · 2.59 FPS)", zh: "bicycle 自行车 (613 万高斯)" },
      hudL: { en: "<strong>Mip-NeRF 360 bicycle</strong> · 6,131,954 SH3 Splats · 4 TPU v5p Chips", zh: "<strong>Mip-NeRF 360 bicycle</strong> · 613 万全 SH3 高斯 · 4 芯 TPU v5p" },
      hudR: { en: "Native <strong>4K (3840x2160) @ 2.59 FPS</strong> (386 ms) · Click to Zoom", zh: "原生 <strong>4K (3840x2160) 2.59 FPS</strong> (386 ms) · 点击放大" }
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
      plateId: "robotwin_bimanual",
      tabIdx: { en: "ROBOT 01 · BIMANUAL", zh: "机器人 01 · 双臂操控" },
      tabTitle: { en: "RoboTwin 16-DOF (123.9k FPS)", zh: "RoboTwin 双臂 (12.39万 FPS)" },
      hudL: { en: "<strong>RoboTwin Bimanual Tabletop</strong> · 72 Geoms · Isaac Replicator 5-Modality DR", zh: "<strong>RoboTwin 16 自由度双臂桌面</strong> · 72 刚体 · Replicator 五模态域随机" },
      hudR: { en: "<strong>123,858 cam-FPS</strong> (41,286 env/s, +0.32% DR) · Click to Zoom", zh: "4 芯实测 <strong>123,858 cam-FPS</strong>（DR 仅 +0.32%）· 点击放大" }
    },
    {
      plateId: "menagerie_g1",
      tabIdx: { en: "ROBOT 02 · HUMANOID", zh: "机器人 02 · 双足人形" },
      tabTitle: { en: "Unitree G1 (393k · 2,546 FPS)", zh: "宇树 G1 人形 (2,546 FPS)" },
      hudL: { en: "<strong>Unitree G1 Humanoid</strong> · 393k Mesh Tris · Wide-BVH-64 + 1024² GGX", zh: "<strong>宇树 Unitree G1 人形机器人</strong> · 39.3 万面 · Wide-BVH-64" },
      hudR: { en: "<strong>2,546 FPS</strong> @ 64² (4 Chips, host_callbacks=0) · Click to Zoom", zh: "4 芯 64² 实测 <strong>2,546 FPS</strong>（零回调）· 点击放大" }
    },
    {
      plateId: "menagerie_go2",
      tabIdx: { en: "ROBOT 03 · QUADRUPED", zh: "机器人 03 · 四足机器狗" },
      tabTitle: { en: "Unitree Go2 (398k · 1,863 FPS)", zh: "宇树 Go2 四足 (1,863 FPS)" },
      hudL: { en: "<strong>Unitree Go2 Quadruped</strong> · 398k Mesh Tris · Curved Chassis + Soft Shadow", zh: "<strong>宇树 Unitree Go2 四足机器狗</strong> · 39.8 万面 · 软阴影 PBR" },
      hudR: { en: "<strong>1,863 FPS</strong> @ 64² (4 Chips, gather_ops=0) · Click to Zoom", zh: "4 芯 64² 实测 <strong>1,863 FPS</strong>（零 Gather）· 点击放大" }
    },
    {
      plateId: "menagerie_panda",
      tabIdx: { en: "ROBOT 04 · 7-DOF ARM", zh: "机器人 04 · 7 轴机械臂" },
      tabTitle: { en: "Franka Panda (3,131 FPS)", zh: "Franka Panda 机械臂 (3,131 FPS)" },
      hudL: { en: "<strong>Franka Emika Panda</strong> · Menagerie MJCF · Diff Physics Servoing Target", zh: "<strong>Franka Emika Panda</strong> · 官方 Menagerie · 可微视觉伺服平台" },
      hudR: { en: "<strong>3,131 FPS</strong> @ 64² (4 Chips) · <code>jax.grad</code> 36.08 dB · Click to Zoom", zh: "4 芯 64² <strong>3,131 FPS</strong> · <code>jax.grad</code> 伺服 36.08 dB · 点击放大" }
    },
    {
      plateId: "menagerie_aerohand",
      tabIdx: { en: "ROBOT 05 · DEXTEROUS", zh: "机器人 05 · 多指灵巧手" },
      tabTitle: { en: "Aero Hand (4,284 FPS)", zh: "Aero Hand 灵巧手 (4,284 FPS)" },
      hudL: { en: "<strong>Tetheria Aero Hand</strong> · Articulated Fingers · Tactile &amp; Manipulation Ready", zh: "<strong>Tetheria Aero Hand 灵巧手</strong> · 多指关节 · 视触觉策略高速底座" },
      hudR: { en: "<strong>4,284 FPS</strong> @ 64² (4 Chips) · Click to Zoom", zh: "4 芯 64² 实测 <strong>4,284 FPS</strong> · 点击放大" }
    }
  ],

  heroStage: [
    {
      video: "./assets/videos/robowild_4dgs_push_4k.mp4",
      poster: "./assets/videos/robowild_4dgs_push_4k_poster.jpg",
      title: {
        en: "RoboWild 4DGS × MJX Panda Pushing LEGO Bulldozer (Native 4K)",
        zh: "RoboWild 4DGS × MJX 机械臂推乐高推土机（原生 4K）"
      },
      hudL: {
        en: "<strong>LIVE NATIVE 4K TPU VIDEO · 4DGS × MJX</strong> · 7-DOF Panda Physically Pushing 125.6k-Splat LEGO Dozer",
        zh: "<strong>原生 4K TPU 实时录屏 · 4DGS × MJX</strong> · 7 轴 Panda 机械臂实时触碰推挤 12.56 万高斯乐高推土机"
      },
      hudR: {
        en: "<strong>3840×2160 Native 4K</strong> · Physics + 3DGS + Shadows in Single XLA Graph",
        zh: "<strong>3840×2160 原生 4K</strong> · 物理+光栅+阴影同驻单 XLA 计算图"
      }
    },
    {
      plateId: "moana_mjx_beach",
      hudL: {
        en: "<strong>Physical AI × Disney Moana OpenUSD</strong> · 22.30B Tris + 45 Live MJX Geoms in Single TLAS",
        zh: "<strong>具身智能 × 迪士尼 Moana OpenUSD</strong> · 223 亿面与 45 个 MJX 关节刚体同图互影"
      },
      hudR: {
        en: "Native <strong>4K (3840x2160)</strong> in <strong>5.40 s</strong> · Click frame for Lightbox + 1:1 Loupe",
        zh: "原生 <strong>4K (3840x2160)</strong> 单帧 <strong>5.40 s</strong> · 点击画面打开 1:1 放大镜"
      }
    },
    {
      plateId: "bistro_humanoid_4k",
      hudL: {
        en: "<strong>Real-to-Sim 3DGS × MJX</strong> · 10M-Gaussian Amazon Bistro + Humanoid &amp; Mobile Panda",
        zh: "<strong>实景 Real-to-Sim 3DGS × MJX</strong> · 1,000 万 Bistro 街区 + 人形与熊猫机器人"
      },
      hudR: {
        en: "Native <strong>4K @ 2.28 FPS</strong> (439.3 ms, gather_ops=0) · Click to Inspect",
        zh: "原生 <strong>4K 2.28 FPS</strong>（439.3 ms，零 HBM Gather）· 点击放大检视"
      }
    },
    {
      plateId: "barcelona_tpu",
      hudL: {
        en: "<strong>Blender Open Data Barcelona Pavilion</strong> · 54.91M Unique Tris Compiled from .blend",
        zh: "<strong>Blender 官方巴塞罗那展馆</strong> · 5,491 万唯一三角面从 .blend 直编上 TPU"
      },
      hudR: {
        en: "Native <strong>4K @ 1.11 img/s</strong> · <strong>15.94 mm</strong> Z-MAE · Click to Inspect",
        zh: "原生 <strong>4K 1.11 img/s</strong> · 几何深度差仅 <strong>15.94 mm</strong> · 点击放大检视"
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
          id: "barcelona",
          tab: { en: "Barcelona 54.9M", zh: "巴塞罗那馆 5,491 万面" },
          aImg: "./assets/compare/barcelona_a.webp",
          bImg: "./assets/compare/barcelona_b.webp",
          aLbl: { en: "LEFT: TPU-RTX (54.91M Tris, 15.94 mm Z-MAE)", zh: "左：TPU-RTX (5,491 万面, 深度差 15.94 mm)" },
          bLbl: { en: "RIGHT: Official Blender Cycles Reference EXR", zh: "右：官方 Blender Cycles 基准真值 EXR" }
        },
        {
          id: "bmw",
          tab: { en: "BMW27 (37.65 dB)", zh: "BMW27 跑车 (37.65 dB)" },
          aImg: "./assets/compare/bmw_a.webp",
          bImg: "./assets/compare/bmw_b.webp",
          aLbl: { en: "LEFT: TPU-RTX Wide-BVH-64 + AgX (0.9813 SSIM)", zh: "左：TPU-RTX Wide-BVH-64 + AgX (SSIM 0.9813)" },
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
          id: "denoise",
          tab: { en: "1-spp -> 4K MXU Denoiser", zh: "1-spp 噪点 -> 4K 降噪 (87.8 FPS)" },
          aImg: "./assets/compare/denoise_a.webp",
          bImg: "./assets/compare/denoise_b.webp",
          aLbl: { en: "LEFT: Raw 1-spp 4-Bounce Monte Carlo (9.70 dB)", zh: "左：原始 1-spp 4 次反弹路径追踪噪点 (9.70 dB)" },
          bLbl: { en: "RIGHT: BF16 MXU 4K Denoiser (24.92 dB, 11.39 ms / 87.8 FPS)", zh: "右：BF16 MXU 4K 降噪输出 (24.92 dB, 11.39 ms)" }
        },
        {
          id: "dlss",
          tab: { en: "TPU-DLSS 3.5 (1080p -> 4K)", zh: "TPU-DLSS 3.5 (58.9 FPS 4K)" },
          aImg: "./assets/compare/dlss_a.webp",
          bImg: "./assets/compare/dlss_b.webp",
          aLbl: { en: "LEFT: Noisy 1080p 1-spp Input Frame (w/ Crop Box)", zh: "左：1080p 1-spp 原始噪点输入（含局部放大框）" },
          bLbl: { en: "RIGHT: Native 4K TPU-DLSS 3.5 Output (58.93 FPS, 3.91x Speedup)", zh: "右：原生 4K TPU-DLSS 3.5 输出（58.93 FPS，3.91 倍提速）" }
        },
        {
          id: "material",
          tab: { en: "Matte Clay -> Chrome (0 RT)", zh: "哑光泥塑 -> 镜面铬合金 (0 光追)" },
          aImg: "./assets/compare/material_a.webp",
          bImg: "./assets/compare/material_b.webp",
          aLbl: { en: "LEFT: Matte Clay (rough=0.88 · w/ MC GT Inset)", zh: "左：哑光泥塑材质（零光追 · 含路径追踪参考小窗）" },
          bLbl: { en: "RIGHT: Brushed Chrome (rough=0.03, metal=1.00, 78.6 FPS 4K)", zh: "右：高光铬合金（原生 4K BF16 实时 78.6 FPS）" }
        },
        {
          id: "tod",
          tab: { en: "Sunset 2600K -> Twilight 9500K", zh: "日落 2600K -> 暮蓝 9500K" },
          aImg: "./assets/compare/tod_a.webp",
          bImg: "./assets/compare/tod_b.webp",
          aLbl: { en: "LEFT: Golden Sunset (2600K · w/ MC GT Inset)", zh: "左：黄金日落暖光 (2600K · 含路径追踪参考小窗)" },
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
      pageTitle: "TPU-RTX — Turning Google TPUs into Real-Time Ray-Tracing Engines | Feiyang Liu",
      themeNight: "Obsidian",
      themeDay: "Paper",
      nav: {
        c1: "01 GPU vs TPU", c2: "02 Software & Bench", c3: "03 Visual Proof",
        c4: "04 Perf & 32-Chip", c5: "05 Embodied AI", c6: "06 Limits & Data"
      },
      hero: {
        kicker: "GOOGLE CLOUD TPU v5p (1-CHIP TO 32-CHIP POD) · ZERO HOST_CALLBACK · ZERO HBM GATHER",
        by: "By",
        license: "Apache-2.0",
        title: "<em>TPU-RTX</em>: Turning Google TPUs into Real&#8209;Time Ray&#8209;Tracing Engines.",
        lead: "Native ray tracing, path tracing, <strong>1.06B</strong>-splat 3DGS, and <strong>17.18B</strong>-triangle OpenUSD / Physical AI on <strong>Google Cloud TPU v5p</strong> — compiled into one zero-copy XLA graph with zero CPU round-trips.",
        ctaStart: "Explore Visual Proof",
        ctaGallery: "Software & Benchmarks",
        stamp: "17 Live 4K/HD TPU Videos + 26 Uncropped Cinema Plates · 100% TPU v5p Verified",
        t0i: "SCENE 01 · LIVE 4K VIDEO",
        t0t: "4DGS × MJX Panda Pushing LEGO Dozer (4K)",
        t1i: "SCENE 02 · FILM × PHYSICAL AI",
        t1t: "Live MJX Robots on 22.30B-Tri Moana Beach (4K)",
        t2i: "SCENE 03 · REAL-TO-SIM 3DGS",
        t2t: "Live MJX Robots in 10M-Splat Amazon Bistro (4K)",
        t3i: "SCENE 04 · BLENDER CYCLES",
        t3t: "Barcelona Pavilion (54.91M Tris, 4K)",
        stackTag: "SYSTEM STACK OVERVIEW · 10-SECOND MAP",
        stackTitle: "How TPU-RTX Connects TPU v5p Silicon to Robotics, OpenUSD, Blender & 3DGS",
        stackSub: "Four clean layers inside one zero-copy XLA HLO graph: TPU v5p Silicon -> 3 Core Kernels -> 8 Drop-In Software Adapters -> Verified Benchmarks."
      },
      kpis: [
        ["End-to-End MJX + Raytrace", "411,023 FPS", "<strong>32-Chip Pod</strong> (<code>B=131k</code>, 7.97x scaling) · <strong>266k FPS</strong> (4c fused)"],
        ["Scene Scale (Moana & Unique)", "22.3B / 17.2B Tris", "Moana OpenUSD (<code>4.65 GiB</code>) &amp; 32c non-instanced tris (<code>1.18 TB HBM</code>)"],
        ["Native 3DGS Speed & Pod Splats", "952k FPS / 1.06B", "<strong>952,248 FPS</strong> (32c 128²) · <strong>1.062B splats</strong> distributed over ICI"],
        ["LiDAR & Cluster HBM Capacity", "63.9B/s · 3.06 TB", "32c <code>128×2048</code> LiDAR (<strong>243.7k env-FPS</strong>) · Single XLA Graph"]
      ],
      c1: {
        num: "01 / 06 · WHY TPU?",
        rail: "First principles: how SIMT RTX GPUs and Systolic TPU MXUs execute ray tracing differently — and where each wins or loses.",
        title: "Why <em>TPU</em>? GPU vs. TPU Silicon &amp; Tradeoffs.",
        lead: "A GPU (SIMT Warps + Hardware RT Cores + L1/L2 Caches) and a Google Cloud TPU v5p (<strong>128×128 Systolic MXU</strong> at 459 TFLOP/s BF16 + <strong>64 MiB software-managed VMEM</strong> + <strong>95.74 GiB HBM</strong>) sit on opposite ends of silicon architecture. Pointer chasing stalls on TPU gather latency; aligning 64-ary Wide-BVH traversal with VMEM turns <code>3.064 TB</code> Pod HBM and MXUs into a unified Physical AI &amp; rendering superpower.",
        gvtTag: "SILICON & DATAFLOW COMPARISON",
        gvtTitle: "SIMT RTX GPU vs. Systolic Google TPU v5p Execution Pipeline",
        gvtSub: "Why CUDA-style pointer chasing stalls on TPU HBM, and how VMEM 64-ary Wide-BVH unlocks systolic ray tracing.",
        gvtRows: [
          ["1. Dynamic Branching & Hardware RT Cores", "<span class='badge fail'>Con</span> No HW RT Cores; <code>128</code>-lane VPU prefers regular, branchless masking.", "<span class='badge ok'>Pro</span> Dedicated hardware RT Cores &amp; 32-thread SIMT warps handle incoherent multi-bounce branching naturally."],
          ["2. Random Memory Gather / Scatter", "<span class='badge fail'>Con</span> High <code>166 FLOP/B</code> ridge point; uncoalesced HBM gathers or tile atomics crater speed.", "<span class='badge ok'>Pro</span> Hardware L1/L2 caches and fast shared-memory atomics absorb irregular pointer chasing."],
          ["3. On-Chip VMEM & HBM Capacity", "<span class='badge ok'>Pro</span> Massive <strong>95.74 GiB HBM/chip (3.064 TB / 32c)</strong> holds <strong>1.06B splats</strong> or <strong>22.3B</strong>-tri Moana without PCIe paging.", "<span class='badge fail'>Con</span> Per-GPU VRAM (24–80 GB) caps massive 3DGS/OpenUSD + RL batch concurrency on one node."],
          ["4. Zero-Copy Physics + Render + RL/VLA", "<span class='badge ok'>Pro</span> Physics (<code>mjx.step</code>), raytrace, GelSight, LiDAR, PPO &amp; ViT-Tiny live in <strong>one XLA graph</strong> (<code>411k FPS</code> 32c, <code>-1.84%</code> VLA overhead).", "<span class='badge fail'>Con</span> Often crosses Warp / OptiX / Isaac / PyTorch bridges or copies buffers between engines."],
          ["5. MXU AI Denoising, DLSS & Relighting", "<span class='badge ok'>Pro</span> <code>459 TFLOP/s</code> BF16 MXUs sit directly behind HBM G-buffers: <strong>87.8 FPS</strong> 4K Denoise, <strong>58.9 FPS</strong> 4K DLSS, <strong>78.6 FPS</strong> 4K Relight.", "<span class='badge ok'>Pro</span> Tensor Cores + mature DLSS 3.5 / OptiX OIDN ecosystem work out of the box."],
          ["6. Startup Compile vs. Interactive Viewport", "<span class='badge fail'>Con</span> XLA AOT pays <strong>4–130 s</strong> cold compile upfront; better for compiled training/renders than live CAD.", "<span class='badge ok'>Pro</span> Sub-second CUDA responsiveness + unmatched interactive DCC tooling (Omniverse, Blender, Unreal)."]
        ]
      },
      c2: {
        num: "02 / 06 · SOFTWARE & BENCH",
        rail: "All 8 rebuilt software adapters, 9-row supported benchmark matrix (verified vs. drop-in ready), and 5-line SDK quickstart.",
        title: "<em>Software Retrofits</em> &amp; Supported Benchmarks.",
        lead: "Instead of forcing robotics teams to rewrite tasks for a bespoke toy renderer, we rebuilt standard simulator APIs, scene formats, and rendering SDKs (<strong>MuJoCo/MJX</strong>, <strong>Isaac Replicator</strong>, <strong>Blender Cycles</strong>, <strong>OpenUSD</strong>, <strong>OptiX</strong>, <strong>gsplat</strong>, <strong>DLSS 3.5</strong>, <strong>ReSTIR</strong>) to compile directly into XLA &amp; Pallas Mosaic on TPU.",
        sdkBar: "Quickstart SDK · Unified Entry Point for Physical AI & Rendering",
        ecoTag: "8 DROP-IN ADAPTERS · CLICKABLE GITHUB SOURCE",
        ecoTitle: "Rebuilt Simulators, DCC Compilers & Rendering APIs for Google TPU",
        ecoSub: "Click any file badge to inspect its XLA / Pallas Mosaic implementation inside tpu_mjx_render/adapters/.",
        adapters: [
          ["MuJoCo / MJX & Menagerie", "mjx_universal.py", "Zero-copy compilation from unmodified MjModel, MJCF XML, or Menagerie multi-mesh scenes into Wide-BVH-64 + analytic primitives, plus jax.grad differentiable servoing.", "266k FPS (4c) / 411k FPS (32c) · Menagerie 1.9k–4.3k FPS"],
          ["NVIDIA Isaac Sim & Replicator", "isaac_replicator.py", "Drop-in TiledCamera (RGB/Depth/Normal/Seg/Flow), rep.randomizer.* XLA domain randomization, 128x2048 RTX LiDAR & GelSight tactile kernels in one graph.", "RoboTwin 123.9k cam-FPS (+0.32% DR) · 32c LiDAR 63.88B rays/s"],
          ["Blender 4.x / Cycles .blend", "blend_compiler.py", "Automated Cycles Shader Graph (Principled / Glass / Glossy / SSS) & Depsgraph walker compiling raw .blend scenes directly into TPU ScenePack + AgX.", "BMW27 37.65 dB · Monster 36.98 dB · Barbershop 31.00 dB"],
          ["Pixar / Disney OpenUSD", "openusd_loader.py", "Native .usd / .usda / .usdc / .usdz loader supporting UsdGeom.Mesh, PointInstancer & BasisCurves into 2-level Wide-TLAS-64 on TPU.", "Disney Moana 22.30B tris in 4.65 GiB · 4K 5.13 s / 8K 17.24 s"],
          ["NVIDIA OptiX Python SDK", "tpu_optix.py", "OptiX-style OptixDeviceContext, accelBuild(GAS/IAS), programmable ShaderBindingTable (raygen/miss/closesthit/anyhit) & live cloth BVH refit.", "Cloth Refit 973.3 FPS (0.0 m error) · 17.18B unique tris (32c)"],
          ["gsplat 3DGS / 4DGS Engine", "tpu_gsplat.py", "Drop-in gsplat-style chunked-VMEM EWA rasterizer + 32-view jax.grad training + SE(3) dynamic 4DGS without dynamic HBM gathers.", "142k (4c) / 952k FPS (32c) · 1.062B splats (32c) · 29.2 MPix/s 8K"],
          ["NVIDIA DLSS 3.5 & Denoiser", "tpu_dlss.py", "Analytical 4x4 motion-vector reprojection + YCoCg history clamp + BF16 MXU 2x Super-Resolution, 4K Monte Carlo denoiser & static G-buffer relighting.", "TPU-DLSS 3.5 58.9 FPS @ 4K · Denoiser 87.8 FPS · Relight 78.6 FPS"],
          ["SIGGRAPH ReSTIR DI", "tpu_restir.py", "Streaming M=32 Resampled Importance Sampling (RIS) + spatiotemporal reservoir reuse over thousands of dynamic lights on TPU HBM.", "1,024-Light Courtyard 21.89 dB (+11.61 dB, 7.43x HDR VarRed)"]
        ],
        bmkTag: "SUPPORTED BENCHMARKS · VERIFIED VS. DROP-IN READY",
        bmkTitle: "Robotics, Embodied AI & Graphics Benchmark Coverage",
        bmkSub: "Explicit distinction between directly measured live TPU v5p benchmarks and API-compatible drop-in MJCF / OpenUSD suites.",
        benchSupportRows: [
          ["<strong>RoboTwin</strong> (Bimanual Tabletop)", "<code>isaac_replicator</code> + <code>mjx_universal</code>", "<span class='badge ok'>Verified</span>", "16-DOF dual-arm + 3-cam DR: <strong>123,858 cam-FPS</strong> (41,286 env/s, +0.32% DR overhead)."],
          ["<strong>RoboWild</strong> (Real-to-Sim 3DGS / 4DGS x MJX)", "<code>tpu_gsplat</code> + <code>mjx_universal</code>", "<span class='badge ok'>Verified</span>", "Live Panda pushing 125.6k LEGO Dozer splats in Mip-NeRF <code>kitchen</code> (+16.43 cm, 5.32 FPS 1080p / 1.30 FPS 4K)."],
          ["<strong>MuJoCo Menagerie</strong> (DeepMind Official)", "<code>mjx_universal</code>", "<span class='badge ok'>Verified</span>", "Unmodified MJCFs: Aero Hand <strong>4,284 FPS</strong>, Panda <strong>3,131</strong>, Unitree G1 <strong>2,546</strong>, Go2 <strong>1,863</strong> (4c 64²)."],
          ["<strong>Factory Sim-to-VLA + GelSight Tactile</strong>", "<code>isaac_replicator</code> + <code>tactile_kernel</code>", "<span class='badge ok'>Verified</span>", "1M frames in <strong>142.6 s</strong>, ViT-Tiny VLA <strong>7,143 FPS</strong> (-1.84% overhead), 4-sensor GelSight <strong>270,625/s</strong>."],
          ["<strong>Vision RL</strong> (Cartpole / Panda / Humanoid-21DOF)", "<code>mjx_universal</code> + <code>shard_map</code>", "<span class='badge ok'>Verified</span>", "Zero-copy HBM PPO: Cartpole <strong>656.4k/s</strong> (94.2%), Panda <strong>211.6k/s</strong> (411k 32c E2E), Humanoid <strong>292.0k/s</strong>."],
          ["<strong>Autonomous Surround-View + 128-Beam LiDAR</strong>", "<code>isaac_replicator</code> + <code>tpu_optix</code>", "<span class='badge ok'>Verified</span>", "8x1080p + 2x128x2048 rig (<strong>66.8 cam-FPS</strong>, 3.13e-8 m XYZ) &amp; 32c batch LiDAR (<strong>63.88B rays/s</strong>)."],
          ["<strong>Blender Open Data &amp; Studio</strong> (8 Scenes)", "<code>blend_compiler</code>", "<span class='badge ok'>Verified</span>", "BMW27 <strong>37.65 dB</strong>, Monster <strong>36.98 dB</strong>, Barbershop <strong>31.00 dB</strong>, Barcelona 54.9M, Junkshop / Monk / Fishy Cat / Classroom."],
          ["<strong>Disney Moana OpenUSD + Mip-NeRF 360 &amp; Bistro</strong>", "<code>openusd_loader</code> + <code>tpu_gsplat</code>", "<span class='badge ok'>Verified</span>", "Moana <strong>22.30B tris</strong> (4K Daylight/Sunset/8K/MJX), 4x Mip-NeRF 360 4K (8K garden <strong>29.2 MPix/s</strong>), 10M Bistro x MJX."],
          ["<strong>RoboDojo / MuJoCo Playground / ALOHA / ManiSkill</strong>", "<code>mjx_universal</code> + <code>openusd_loader</code> + <code>isaac_replicator</code>", "<span class='badge cyan'>Drop-In Ready</span>", "Standard MJCF XML, OpenUSD .usd &amp; rep.randomizer APIs compile directly via <code>tpu_rtx.compile(...)</code> (benchmark runs not yet recorded in out/metrics/)."]
        ]
      },
      c3: {
        num: "03 / 06 · VISUAL PROOF",
        rail: "11 real-time 4K/HD TPU videos + 6 Cycles split sliders, 8 optics sliders, 5 4K/8K plates, 5 4K 3DGS stages & 5 neural sliders.",
        title: "<em>Visual Proof</em>: Live 4K Videos, Film &amp; Real&#8209;World 3DGS.",
        lead: "Explore <strong>11 real-time 30 fps TPU videos</strong> (Native 4K 3DGS/4DGS orbits, 4K <code>f/1.4</code> thin-lens focus pulls, live PBD cloth BVH refit, and zero-raytrace MXU relighting), alongside uncropped Blender Cycles sliders, physical-optics comparisons, and 4K/8K cinema plates."
      },
      c4: {
        num: "04 / 06 · PERF & 32-CHIP",
        rail: "Verified 32-chip (8-host, 3.064 TB HBM) scaling highlights plus 5 full-width profiling charts (no hidden negative runs).",
        title: "<em>587×</em> Optimization &amp; 32&#8209;Chip Pod Scaling (3.06 TB HBM).",
        lead: "Every number from 1 chip to <strong>32 TPU v5p chips</strong> (<code>3.064 TB HBM</code>, full capstone completed live in <code>395.07 s</code>) is backed by raw telemetry. Below are our 4 pod scaling breakthroughs followed by 5 full-width profiling plots — including negative/flat experiments and the <code>64 MiB</code> VMEM cliff."
      },
      c5: {
        num: "05 / 06 · EMBODIED AI",
        rail: "Four live TPU convergence & throughput plots + 5-tab PBR robot switcher (RoboTwin Bimanual, G1, Go2, Panda, Aero Hand).",
        title: "In&#8209;HBM Training: <em>Diff Physics</em>, Vision PPO &amp; Robots.",
        lead: "Physics steps, tactile/LiDAR sensors, losses, and gradients train together inside TPU HBM with zero CPU round-trips. Below are four exact telemetry curves plus an interactive 5-robot PBR stage covering <strong>RoboTwin Bimanual</strong> and official <strong>MuJoCo Menagerie</strong> models."
      },
      c6: {
        num: "06 / 06 · LIMITS & DATA",
        rail: "Real TPU post-mortems, QA-rejected failures, honest comparison citing wenhaochai/rtx-h100, 21-row JSON ledger, plate index & BibTeX.",
        title: "Post&#8209;Mortems, <em>Honest Boundaries</em> &amp; 21&#8209;Row Ledger.",
        lead: "Four candid engineering post-mortems, two uncropped QA-rejected failure renders, an honest hardware comparison citing <code>wenhaochai/rtx-h100</code>, an interactive telemetry console, all <strong>21 traceable JSON logs</strong>, and a quick-jump directory across all <strong>26 clean plates</strong>.",
        pmCards: [
          ["FAIL -> FIX", "Checkerboard Gather Bottleneck", "1. Why Early Shading Ate >90% of Frame Time (151 MB Gather)", "Amdahl profiling revealed >90% of early Pallas frame time (286.5 ms) wasn't in the BVH kernel — it was a single XLA dynamic gather geom_xmat[b_idx, safe_gid] triggered by one checkerboard floor plane.", "Replacing that dynamic lookup with static iteration over checkerboard IDs slashed shading 4.4x (286.5 -> 64.9 ms), unlocking 12,843 FPS (1c) and 50,131 FPS (4c) with gather_ops = 0."],
          ["NEGATIVE RESULT", "3DGS Pallas vs XLA Baseline", "2. Hand-Written Pallas 3DGS Lost 1.81x to Pure XLA", "Our initial MXU log-space Pallas 3DGS kernel (gs3d_pallas) ran at only 658.6 FPS — slower than plain XLA jax_tile (1,193.2 FPS) because padded (BT, 64, 128) outputs bloated writes 16x and emitted 65,536 tiny DMAs.", "We documented the negative result, kept jax_tile default, then redesigned chunked VMEM EWA sorting (gs3d_v3) to jump 119x to 142,008 FPS (4c) and 952,248 FPS + 1.062B splats (32c)."],
          ["3-RUN FAILURE -> PASS", "Neural Shader Gate Ablation", "3. Why Stage-2 Neural Relighting Failed Three Times First", "Pixel-only MLP failed soft penumbras (27.80 dB); narrow W32 U-Net failed gripper occlusion (10% mismatch); jnp.clip inside loss saturated gradients on step 1 leaving training stuck at 15.20 dB.", "Adding reflect(V, N) + Fresnel N dot V, dual 1x1 BRDF skip heads, unclipped loss + global norm clipping, and tight-grasp augmentation passed 8/8 locked gates at 38.35 dB / 0.9889 SSIM."],
          ["MULTI-HOST PITFALL -> SOLVED", "32-Chip Pod 395.1 s", "4. Non-Addressable MJX Arrays & SSH PID Lock Leaks", "In 8-host (32-chip) JAX distributed mode, mjx.put_model(m, impl='jax') with device=None created cross-host IFRT failures, and multiline SSH PID cleanup previously left /dev/vfio/3 locked.", "Patched device=local_devs[0] + single-line tr PID cleanup, then executed the live 32-chip (3.064 TB HBM) Capstone across all 8 hosts in 395.07 s: 411k E2E, 952k 3DGS, 1.062B splats, 63.88B LiDAR rays/s, 17.18B unique tris & 1,650 spm @ 8K, returning 8/8 IDLE."]
        ],
        ablationRows: [
          ["Locked Phase-0 Threshold", ">= 30.0 dB", ">= 28.0 dB", "<= 0.010", "<span class='badge cyan'>All 8 Required</span>"],
          ["Stage 1 Direct Exact Baseline", "13.83 dB", "—", "—", "Hard shadows, no indirect GI"],
          ["M1 Pixel MLP (1x1, 150 ep)", "27.80 dB", "23.52 dB", "0.0500 (7/140)", "<span class='badge fail'>FAIL</span> Zero spatial receptive field"],
          ["M2 Spatial U-Net (Narrow W32)", "35.58 dB", "27.79 dB", "0.1000 (14/140)", "<span class='badge fail'>FAIL</span> Grasping contact blur"],
          ["M2 Spatial U-Net + jnp.clip Loss", "15.20 dB", "—", "—", "<span class='badge fail'>FAIL</span> Dead step-1 gradient saturation"],
          ["M2 Spatial U-Net (24-ch BRDF, Final)", "<strong>38.35 dB</strong>", "<strong>30.31 dB</strong>", "<strong>0.00714 (1/140)</strong>", "<span class='badge ok'>PASS (8/8)</span> SSIM 0.9889, Edge 0.157 px"]
        ],
        rejected: [
          {
            id: "rejected_openusd_clay",
            img: "./assets/postmortem/rejected_openusd_clay.webp",
            banner: "QA REJECTED · EARLY FLAT CLAY SHADING & LOW MATERIAL CONTRAST",
            title: "Rejected Output A: Early Untextured OpenUSD Stage Test (v3_w2_openusd_4k)",
            desc: "Caught by Visual QA Gate: before binding full Disney OpenUSD Ptex/material packs, diffuse shaders collapsed distant foliage and water reflections into flat clay. Replaced by the verified Daylight, Sunset, and Moana x MJX 4K passes."
          },
          {
            id: "rejected_garden_seams",
            img: "./assets/postmortem/rejected_garden_seams.webp",
            banner: "QA REJECTED · VISIBLE 16x16 TILE SEAMS & BLACK BACKGROUND",
            title: "Rejected Output B: Early 32-View 3DGS Training Render (v3_w3_garden_32v_grad)",
            desc: "Caught by Visual QA Gate: local tile accumulation bounds clipped splats outside each 16x16 frustum bin, producing blocky grid seams and masked black borders. Excluded from showcase galleries; fixed fast-grad pipeline verified at 31.31 dB peak."
          }
        ],
        honest: "Every number reported on this page — across 1-chip experiments, single-host 4-chip nodes (383 GiB HBM), and the full 8-host 32-chip TPU v5p-64 Pod (3.064 TB HBM, v3_pod_v5p64_capstone.json) — was measured live on real Google Cloud TPU v5p silicon with zero fabricated extrapolation. Standard suites without a benchmark JSON yet (RoboDojo, MuJoCo Playground, ALOHA, ManiSkill) are explicitly labeled Drop-In Ready rather than verified scores. Meanwhile, GPUs still lead decisively in sub-second interactive editor responsiveness, dynamic incoherent branching on HW RT Cores, and single-card Blender Cycles speed (Chai et al., 2026).",
        authorBio: "Systems & graphics engineer exploring compiler-native rendering, differentiable physics, and high-throughput Physical AI on custom ML accelerators.",
        ack: "Third-Party Asset & Work Acknowledgments: Walt Disney Animation Studios Moana Island Scene v2.1 (Non-Commercial License); Blender Open Data / Blender Studio (BMW27, Barcelona Pavilion, Barbershop, Classroom, Junkshop, Lone Monk, Monster Under the Bed, Fishy Cat — CC0 / CC-BY); Mip-NeRF 360 photogrammetry captures (Barron et al.); Google DeepMind mujoco_menagerie (Apache-2.0 / BSD); Chai et al. (2026) wenhaochai/rtx-h100 Cycles H100/RTX PRO 6000 reference."
      },
      podStats: [
        ["Bench 1 & 2 · Embodied Physics + 3DGS (4c -> 32c)", "411,023 FPS E2E  |  952,248 FPS 3DGS", "7.97x pod scaling from 51.5k MJX+RTX (4c) to 411.0k FPS (32c B=131k); 128² 3DGS hits 952.2k FPS plus 1.062B ICI splats @ 1080p (3.41 s)."],
        ["Bench 3 · 128x2048 Batch 360° LiDAR (32c)", "243,676 env-FPS  |  63.88 Billion Rays/s", "B=16,384 parallel 128x2048 LiDAR scans in 67.2 ms across 8 hosts (B=32,768: 229,060 env-FPS / 60.05B rays/s, gather_ops=0)."],
        ["Bench 4 · Geometry-Sharded Unique Triangles (32c)", "17.18B Unique Non-Instanced Tris (1.18 TB)", "32-chip ICI jax.lax.pmin scales 8x over 4c (2.15B -> 17.18B unique tris), rendering 4K in 4.46 s and 8K in 8.53 s (8.59B tris: 2.43 s 4K / 4.70 s 8K)."],
        ["Bench 5 · Spatial-Sharded Native 8K 3-Bounce PT (32c)", "1,650.2 spm @ 8K (5.475 GRays/s, 7.54x 4c)", "Unbiased 3-bounce Monte Carlo GI @ 7680x4320 (33.18M px): 145.4 ms @ 4 spp (6.88 FPS) and 37.5 ms @ 1 spp (26.69 FPS, host_callbacks=0)."]
      ],
      compRows: [
        ["Embodied Vision RL, VLA & Tactile", "<strong>Zero-Copy Single XLA Graph</strong>: <code>266k FPS</code> (4c) / <code>411k FPS</code> (32c), <code>1M</code> factory frames in <code>142.6s</code>, <code>270.6k/s</code> GelSight", "Static offline renderer (no interactive RL loop)", "Mature CUDA Warp/Isaac ecosystem on RTX/L40S/H100"],
        ["Real-to-Sim 3DGS / 4DGS + Physics", "<strong>Zero-Gather Native TPU</strong>: <code>142k</code> (4c) / <code>952k</code> (32c), <code>1.062B</code> distributed splats, <code>8K garden 29.2 MPix/s</code>, live <code>4DGS x MJX</code> push", "Cycles mesh + NRC only", "Requires separate CUDA gsplat + USD/PhysX bridge"],
        ["End-to-End Diff Physics + Render", "<strong>36.08 dB / 1.63 cm EE Error</strong> via <code>jax.value_and_grad(mjx.step + PBR)</code> at <code>27.9 grad-steps/s</code>", "Non-differentiable offline path tracer", "Separate Warp diff-sim / Nvdiffrast integration"],
        ["Film Production Path Tracing & Cycles", "<strong>Large-Scene OpenUSD & .blend Compiler</strong>: <em>Moana</em> (<code>22.30B</code> tris, 4.65 GiB), <code>BMW27 37.65 dB</code>, <code>17.18B</code> unique tris (32c)", "<strong>Speed Champion (wenhaochai/rtx-h100)</strong>: <code>55,148 spm</code> single H100 (7.8x Blender), <code>78,297 spm</code> RTX PRO 6000 OptiX (39.9 img/s NRC)", "Full interactive Omniverse / OptiX workstation viewport"],
        ["AI Relighting, Denoising & DLSS", "<strong>Native MXU BF16/INT8</strong>: 4K Denoise <code>87.8 FPS</code>, 4K DLSS <code>58.9 FPS</code>, Static Relight <code>78.6 FPS</code> 4K / <code>307 FPS</code> 1080p", "NRC-scatter <code>40.4 img/s</code> (32-spp bar, H100)", "Hardware Tensor Core DLSS 3.5 + Ray Reconstruction"]
      ],
      ui: {
        copy: "Copy", copied: "Copied", zoom4k: "Inspect ↗", loupe: "1:1 Pixel Loupe (L)", raw: "Open Full Image ↗",
        footerL: "TPU-RTX · Google Cloud TPU v5p (JAX + Pallas Mosaic · Zero host_callbacks · Zero HBM Gathers)",
        searchPlaceholder: "Search experiments, scenes, JSON...",
        v3_head: "TPU-RTX Real-Time Rendered Video Cinema (Native 4K · 30 fps)",
        v5_head: "Embodied Robotics, Diff-Physics Servoing &amp; Menagerie Motion (30 fps)",
        vid_all: "All Videos (11)", vid_gs3d: "4K 3DGS / 4DGS (6)", vid_optics: "4K Bokeh & Cloth (3)", vid_relight: "Neural Relight (2)",
        b_mjx: "Robotics & Embodied AI", b_gs3d: "3DGS / 4DGS / 1.06B Pod", b_cinema: "Moana, Cloth & Blender", b_dlss: "TPU-DLSS 3.5 & Relight", b_fx: "Path Tracing & Optics",
        g_all: "All Standalone Plates (26)", g_film: "Moana & Blender Studio (10)", g_optics: "Physical Optics (3)", g_gs3d: "Real-World 3DGS / Bistro (6)", g_robotics: "Menagerie & RoboTwin (5)", g_rej: "QA-Rejected (2)",
        th1: "Workload", th2: "TPU-RTX (Google TPU v5p)", th3: "GPU Cycles (wenhaochai/rtx-h100)", th4: "NVIDIA Isaac Sim / OptiX",
        gvt1: "Dimension", gvt2: "Google Cloud TPU v5p (TPU-RTX)", gvt3: "NVIDIA GPU (RTX / H100 / Isaac)",
        bs1: "Benchmark / Suite", bs2: "Adapter Stack", bs3: "Status", bs4: "Measured Result or Compatibility Scope",
        ab1: "Stage-2 Neural Shader Ablation", ab2: "PSNR (dB)", ab3: "ROI Contact PSNR", ab4: "Occ Mismatch", ab5: "Gate & Root Cause",
        ld_all: "All (21)", ld_rob: "Robotics", ld_film: "Film & Cloth", ld_gs: "3DGS/4DGS", ld_nn: "Neural/Optics",
        lt1: "Experiment", lt2: "Config / Scale", lt3: "Throughput", lt4: "Accuracy / Parity", lt5: "JSON",
        ev_t: "CHRONOLOGICAL STEP-ENVELOPE · NEGATIVE RUNS EXPOSED", ev_h: "587x Optimization Trajectory on TPU v5p (453 -> 266,209 FPS · 32c: 411,023 FPS)", ev_s: "Every throughput step plus negative/flat runs plotted cleanly (inspired by wenhaochai/rtx-h100).",
        gs_t: "LOG-SCALE PIVOT · 3D GAUSSIAN SPLATTING", gs_h: "3DGS Trajectory: Negative Pallas (-1.81x) -> 142k (4c) -> 952k FPS (32c Pod)", gs_s: "Replacing padded Pallas DMA writes with chunked VMEM EWA unlocks 142k (4c) and 952,248 FPS + 1.062B splats (32c Pod).",
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
      pageTitle: "TPU-RTX — 把 Google TPU 变成原生实时光追引擎 | Feiyang Liu",
      themeNight: "暗房",
      themeDay: "纸本",
      nav: {
        c1: "01 GPU对比TPU", c2: "02 软件与Bench", c3: "03 画质与拉杆",
        c4: "04 性能与32芯", c5: "05 具身训练", c6: "06 踩坑与数据"
      },
      hero: {
        kicker: "GOOGLE CLOUD TPU v5p 真机实测（单芯至 32 芯集群）· 零 HOST_CALLBACK · 零 HBM GATHER",
        by: "作者",
        license: "Apache-2.0 开源",
        title: "<em>TPU-RTX</em>：把 Google TPU 变成原生实时光追引擎。",
        lead: "全栈原生运行于 <strong>Google Cloud TPU v5p</strong>：单张零拷贝 XLA 计算图直驱 <strong>223 亿</strong>面电影 OpenUSD、<strong>10.6 亿</strong> 3DGS、<strong>171.8 亿</strong>唯一三角面与 <strong>41.1 万 FPS</strong> 具身物理仿真。",
        ctaStart: "先看真实画质对比",
        ctaGallery: "支持的软件与 Bench",
        stamp: "17 段真机实时 TPU 视频 + 26 张 4K/8K 纯净大图 · 100% TPU v5p 实测",
        t0i: "场景 01 · 原生 4K 实拍视频",
        t0t: "4DGS × MJX 机械臂推 4DGS 推土机 (4K)",
        t1i: "场景 02 · 电影 × 具身",
        t1t: "MJX 机器人奔跑在 223 亿面 Moana 沙滩（4K）",
        t2i: "场景 03 · 实景 3DGS × 具身",
        t2t: "MJX 人形与熊猫进入 1,000 万 Bistro 街区（4K）",
        t3i: "场景 04 · BLENDER CYCLES",
        t3t: "巴塞罗那展馆（5,491 万唯一面，原生 4K）",
        stackTag: "10 秒系统总览图 · 四层全景架构",
        stackTitle: "TPU-RTX 如何把 Google TPU v5p 底层硬件直接连通具身智能、OpenUSD、Blender 与 3DGS",
        stackSub: "从下至上四层全部驻留在同一张零拷贝 XLA 计算图内：TPU v5p 硬件 -> 3 大零拷贝核心内核 -> 8 大软件适配器 -> 已验证基准套件。"
      },
      kpis: [
        ["端到端物理 + 光追", "411,023 FPS", "<strong>32 芯集群实测</strong>（<code>B=131k</code>，7.97x 扩展）· 4 芯融合 <strong>266k FPS</strong>"],
        ["超大场景几何上限", "223 亿 / 172 亿", "Moana OpenUSD（<code>4.65 GiB</code>）&amp; 32 芯非实例化独立三角面（<code>1.18 TB HBM</code>）"],
        ["3DGS 速度与分布式容量", "95.2 万 FPS / 10.6 亿", "<strong>952,248 FPS</strong>（32 芯 128²）· ICI 跨机分布式 <strong>10.62 亿高斯</strong>"],
        ["激光雷达与整集群显存", "639 亿/s · 3.06 TB", "32 芯 <code>128×2048</code> LiDAR（<strong>24.37 万 env-FPS</strong>）· 全程零回调"]
      ],
      c1: {
        num: "01 / 06 · 为什么用 TPU",
        rail: "第一性原理对比：传统 SIMT 光追 GPU 与 Google TPU 脉动阵列的底层区别，以及各自真正的擅长与短板。",
        title: "为什么用 <em>TPU</em>？GPU 与 TPU 底层架构对比。",
        lead: "传统 GPU（32 线程 SIMT Warp + 硬件 RT Core + 硬件 L1/L2 缓存）与 Google Cloud TPU v5p（459 TFLOP/s BF16 <strong>128×128 脉动阵列 MXU</strong> + <strong>64 MiB 软件显控 VMEM</strong> + <strong>95.74 GiB HBM</strong>）处于芯片设计的两端。照搬 CUDA 随机指针写法会在 TPU Gather 延迟上瞬间卡死；而改用 64 叉 Wide-BVH 对齐 VMEM 后，<code>3.064 TB</code> 大显存和物理-渲染-AI 零拷贝融合成了决定性优势。",
        gvtTag: "硬件架构与执行流程图对比",
        gvtTitle: "传统 SIMT RTX GPU vs. Google TPU v5p 脉动阵列光追流水线对比",
        gvtSub: "拆解两者的核心算力单元、缓存层次、BVH 遍历路径与后续 AI/物理仿真耦合方式。",
        gvtRows: [
          ["1. 动态分支与硬件 RT Core", "<span class='badge fail'>劣势</span> 无专用硬件 RT Core，128 通道 VPU 要求规则无分支掩码。", "<span class='badge ok'>优势</span> 拥有硬件 RT Core 与 32 线程 SIMT Warp，天然擅长随机多反弹光线分歧。"],
          ["2. 随机内存 Gather / 原子操作", "<span class='badge fail'>劣势</span> 脊点高达 166 FLOP/B，一旦出现不规则 HBM Gather 或 Tile 原子写，性能会断崖下跌。", "<span class='badge ok'>优势</span> 硬件 L1/L2 缓存与高速片上原子写专门为随机指针跳转和细粒度 Scatter 设计。"],
          ["3. 显存容量 (HBM) 与片上 VMEM", "<span class='badge ok'>优势</span> 单芯 <strong>95.74 GiB HBM（32 芯 3.064 TB）</strong>轻松装下 <strong>10.6 亿高斯</strong>或 <strong>223 亿面</strong> <em>Moana</em>，无需 PCIe 换页。", "<span class='badge fail'>劣势</span> 单卡显存通常 24–80 GB，大批量具身 RL 或超大规模 3DGS 常碰显存天花板。"],
          ["4. 物理 + 渲染 + RL/VLA 零拷贝融合", "<span class='badge ok'>优势</span> MJX 物理、光追、GelSight、LiDAR、PPO 与 ViT-Tiny 同驻<strong>单张 XLA 计算图</strong>（32 芯 <code>411k FPS</code>，VLA 仅 <code>-1.84%</code> 开销）。", "<span class='badge fail'>劣势</span> 多跨 Warp / OptiX / Isaac / PyTorch 工具链桥接，常伴随中间缓冲拷出。"],
          ["5. MXU 神经降噪、超分与重打光", "<span class='badge ok'>优势</span> <code>459 TFLOP/s</code> BF16 MXU 直连 HBM G-Buffer：<strong>87.8 FPS</strong> 4K 降噪、<strong>58.9 FPS</strong> 4K DLSS、<strong>78.6 FPS</strong> 重打光。", "<span class='badge ok'>优势</span> Tensor Core + 成熟的官方 DLSS 3.5 与 OptiX OIDN 开箱即用。"],
          ["6. 首次编译延迟 vs 交互式视口", "<span class='badge fail'>劣势</span> XLA AOT 需 <strong>4–130 秒</strong>冷编译，更适合预编译训练/渲染而非频繁改拓扑的编辑器。", "<span class='badge ok'>优势</span> 亚秒级响应，加之 Omniverse、Blender、Unreal 无可替代的实时视口生态。"]
        ]
      },
      c2: {
        num: "02 / 06 · 软件改造与 BENCH",
        rail: "8 大已重构软件适配器（点击直达源码）、9 类主流机器人/图形 Bench 覆盖表，以及 5 行接入 SDK。",
        title: "已重构 <em>软件生态</em> 与支持的 Bench 全景。",
        lead: "我们没有另起炉灶要求用户重写工程，而是把行业主流仿真器接口、场景格式和渲染 API（<strong>MuJoCo / MJX</strong>、<strong>Isaac Replicator</strong>、<strong>Blender Cycles</strong>、<strong>OpenUSD</strong>、<strong>OptiX</strong>、<strong>gsplat</strong>、<strong>DLSS 3.5</strong>、<strong>ReSTIR</strong>）直接改造成零拷贝编译进 TPU XLA / Pallas。",
        sdkBar: "极速接入 SDK · tpu_mjx_render.adapters 统一编译入口",
        ecoTag: "8 大 DROP-IN TPU 适配器 · 点击直达 GITHUB 源码",
        ecoTitle: "我们为 Google TPU 重构了哪些仿真器、渲染器与 DCC 管线",
        ecoSub: "点击右上角文件名即可直达 GitHub 仓库 tpu_mjx_render/adapters/ 查看源码实现。",
        adapters: [
          ["MuJoCo / MJX 与官方 Menagerie", "mjx_universal.py", "零改模直接编译官方 MjModel、MJCF XML 与 Menagerie 多网格机器人至 Wide-BVH-64 + 解析几何，并支持穿透 mjx.step 的 jax.grad 可微视觉伺服。", "266k FPS (4芯) / 411k FPS (32芯) · Menagerie 1.9k–4.3k FPS"],
          ["NVIDIA Isaac Sim & Omniverse Replicator", "isaac_replicator.py", "原生重实现 TiledCamera（RGB/Depth/Normal/Seg/Flow）、rep.randomizer.* XLA 域随机化、128x2048 激光雷达与 GelSight 视触觉管线。", "RoboTwin 123.9k cam-FPS (+0.32% DR) · 32芯 LiDAR 63.88B/s"],
          ["Blender 4.x / Cycles .blend 直编器", "blend_compiler.py", "自动递归解析任意 .blend 的 Cycles 材质节点图（Principled/Glass/Glossy/SSS）与 Depsgraph 修改器实例，直编 TPU ScenePack + AgX。", "BMW27 37.65 dB · Monster 36.98 dB · Barbershop 31.00 dB"],
          ["Pixar / Disney OpenUSD 编译加载器", "openusd_loader.py", "原生解析 .usd / .usda / .usdc / .usdz 中的 UsdGeom.Mesh、PointInstancer 与 BasisCurves，直入 TPU 两级 Wide-TLAS-64。", "迪士尼 Moana 223 亿面仅占 4.65 GiB · 4K 5.13s / 8K 17.24s"],
          ["NVIDIA OptiX 等价 Python/JAX SDK", "tpu_optix.py", "在 TPU 上复刻 OptixDeviceContext、accelBuild(GAS/IAS)、可编程 ShaderBindingTable（raygen/miss/closesthit/anyhit）与动态布料 Refit。", "布料 BVH Refit 973.3 FPS (0.0 m 误差) · 32芯 171.8 亿唯一面"],
          ["gsplat 风格 3DGS / 4DGS 引擎", "tpu_gsplat.py", "无动态 HBM Gather 的分块 VMEM EWA 高斯光栅化器：兼容 32 视角 jax.grad 快速训练与 SE(3) 动态 4DGS 物理碰撞。", "14.2万 (4芯) / 95.2万 FPS (32芯) · 10.62 亿高斯 · 原生 8K 实景"],
          ["NVIDIA DLSS 3.5 与光线重建降噪器", "tpu_dlss.py", "精确 4x4 运动矢量重投影 + YCoCg 历史钳位 + BF16 MXU 2x 超分、4K 路径追踪降噪与零光追静态 G-Buffer 交互重打光。", "TPU-DLSS 3.5 4K 58.9 FPS · 4K 降噪 87.8 FPS · 重打光 78.6 FPS"],
          ["SIGGRAPH ReSTIR DI 多光源重采样", "tpu_restir.py", "纯 XLA 流式 M=32 RIS 与时空蓄水池复用，无需 CPU 回传即可应对 1,024 盏动态光源复杂照明。", "1,024 盏灯庭院 21.89 dB（+11.61 dB，HDR 方差降 7.43 倍）"]
        ],
        bmkTag: "基准覆盖矩阵 · 真机实测 vs. 接口直连兼容",
        bmkTitle: "已验证具身智能、机器人与电影级图形 Bench 一览",
        bmkSub: "严格区分已在 TPU v5p 跑出真实记录的 Benchmark 与通过标准 MJCF / OpenUSD 接口即插即用的套件（未实测绝不虚标数字）。",
        benchSupportRows: [
          ["<strong>RoboTwin</strong>（双臂桌面操控与域随机）", "<code>isaac_replicator</code> + <code>mjx_universal</code>", "<span class='badge ok'>真机已测</span>", "16 自由度双臂 + 三相机 Replicator DR：<strong>123,858 cam-FPS</strong>（41,286 env/s，DR 开销仅 +0.32%）。"],
          ["<strong>RoboWild</strong>（实景 3DGS / 4DGS x MJX 交互）", "<code>tpu_gsplat</code> + <code>mjx_universal</code>", "<span class='badge ok'>真机已测</span>", "真实 Mip-NeRF <code>kitchen</code> 内 Panda 实臂撞推 12.56 万乐高推土机高斯（+16.43 cm，1080p 5.32 / 4K 1.30 FPS）。"],
          ["<strong>MuJoCo Menagerie</strong>（DeepMind 官方机器人库）", "<code>mjx_universal</code>", "<span class='badge ok'>真机已测</span>", "未改模型官方 MJCF 直编：Aero Hand <strong>4,284 FPS</strong>、Panda <strong>3,131</strong>、宇树 G1 <strong>2,546</strong>、Go2 <strong>1,863</strong>（4 芯 64²）。"],
          ["<strong>千帧工厂 Sim-to-VLA + GelSight 视触觉</strong>", "<code>isaac_replicator</code> + <code>tactile_kernel</code>", "<span class='badge ok'>真机已测</span>", "<strong>142.6 s</strong> 生成百万帧 VLA，ViT-Tiny 在线训练 <strong>7,143 FPS</strong>（仅 -1.84% 开销），4 路 GelSight <strong>270,625/s</strong>。"],
          ["<strong>零拷贝闭环 Vision RL</strong>（Cartpole/Panda/Humanoid）", "<code>mjx_universal</code> + <code>shard_map</code>", "<span class='badge ok'>真机已测</span>", "像素全片上闭环 PPO：Cartpole <strong>656.4k/s</strong>（94.2%）、Panda <strong>211.6k/s</strong>（32 芯 411k E2E）、Humanoid <strong>292.0k/s</strong>。"],
          ["<strong>自动驾驶 8 路环视 + 双 128 线激光雷达</strong>", "<code>isaac_replicator</code> + <code>tpu_optix</code>", "<span class='badge ok'>真机已测</span>", "8x1080p + 2x128x2048 实测 <strong>66.8 cam-FPS</strong>（3.13e-8 m 误差），32 芯批量 LiDAR <strong>638.8 亿射线/秒</strong>。"],
          ["<strong>Blender Open Data &amp; Studio</strong>（8 大标准场景）", "<code>blend_compiler</code>", "<span class='badge ok'>真机已测</span>", "BMW27 <strong>37.65 dB</strong>、Monster <strong>36.98 dB</strong>、Barbershop <strong>31.00 dB</strong>、Barcelona 5,491 万面及 Junkshop / Monk / Fishy Cat / Classroom。"],
          ["<strong>迪士尼 Moana OpenUSD + Mip-NeRF 360 &amp; Bistro</strong>", "<code>openusd_loader</code> + <code>tpu_gsplat</code>", "<span class='badge ok'>真机已测</span>", "Moana <strong>223 亿面</strong>（4K 晴天/日落/8K/MJX），4 景 Mip-NeRF 360 4K（8K garden <strong>29.2 MPix/s</strong>），千万级 Bistro x MJX。"],
          ["<strong>RoboDojo / MuJoCo Playground / ALOHA / ManiSkill</strong>", "<code>mjx_universal</code> + <code>openusd_loader</code> + <code>isaac_replicator</code>", "<span class='badge cyan'>接口直连兼容</span>", "所有导出为标准 MJCF XML、OpenUSD .usd 或调用 <code>rep.randomizer</code> 的任务均可直接传入 <code>tpu_rtx.compile(...)</code> 运行（<code>out/metrics/</code> 暂未单独登记其跑分）。"]
        ]
      },
      c3: {
        num: "03 / 06 · 视觉画质与动态视频",
        rail: "11 段原生 4K/高清实时视频展厅 + Blender Cycles、波动光学、4K 实景 3DGS 与神经重打光全对比。",
        title: "<em>真实画质检验</em>：原生 4K 实时视频、电影光学与实景 3DGS。",
        lead: "上方新增 <strong>11 段 Google TPU v5p 真机录制的 30fps 视频展厅</strong>（含原生 4K Mip-NeRF 360 环绕、4K 4DGS×MJX 推土机物理交互、原生 4K <code>f/1.4</code> 景深拉焦、可变形丝绸 BVH 重拟合及零光追 MXU 昼夜/材质变化），随后完整保留全部 Blender Cycles 与光学拉杆。"
      },
      c4: {
        num: "04 / 06 · 性能剖析与 32 芯实测",
        rail: "32 芯整集群（8 机 3.064 TB HBM）实测扩展卡片 + 5 张全宽性能剖析大图（公开包括负优化在内的完整过程）。",
        title: "<em>587×</em> 性能演进与 32 芯集群扩展（3.06 TB HBM）。",
        lead: "从单芯到完整 <strong>8 机 32 芯 TPU v5p-64 集群</strong>（<code>3.064 TB HBM</code>，仅 <code>395.07 s</code> 一次跑通全部 5 套极限压测）的每一项数字均有 JSON 原始日志对应；随后 5 张全宽性能图完整保留负优化试验与 <code>64 MiB</code> VMEM 溢出边界。"
      },
      c5: {
        num: "05 / 06 · 具身智能训练与机器人",
        rail: "4 组 TPU 显存内训练收敛与吞吐曲线 + 5 台真实机器人 PBR 大舞台（RoboTwin 双臂、G1、Go2、Panda、Aero Hand）。",
        title: "TPU 显存内训练：<em>可微物理</em>、Vision PPO 与真实机器人。",
        lead: "物理步进、触觉/雷达传感器、损失计算与梯度反传全部同驻 TPU HBM。下方呈现 4 张全宽训练/传感器遥测图与 <strong>RoboTwin 双臂 + MuJoCo Menagerie</strong> PBR 大舞台。"
      },
      c6: {
        num: "06 / 06 · 踩坑、边界与全量数据",
        rail: "4 个真实负优化复盘、2 张被 QA 拦截的淘汰成片、引自 wenhaochai/rtx-h100 的坦诚对照、21 项可溯源 JSON 账本与 26 景目录。",
        title: "<em>工程踩坑</em>、坦诚边界与 21 份溯源账本。",
        lead: "完整公开 4 次真实踩坑/负结果、2 张被 <strong>Visual QA 门禁淘汰</strong>的反面成片、对照 <code>wenhaochai/rtx-h100</code> 的坦诚能力边界、可切换遥测面板、<strong>21 份 GitHub 溯源 JSON 账本</strong>及全部 <strong>26 张无暇单场景</strong>直达目录。",
        pmCards: [
          ["翻车 -> 破局", "棋盘格隐式 Gather 瓶颈", "1. 以为求交慢，结果 90% 耗时卡在 151 MB 棋盘格 Gather", "早期 Amdahl 剖面揪出惊人真相：286.5 ms 帧耗时里 >90% 根本不在 BVH，而在一块棋盘地面触发的动态 HBM gather geom_xmat[b_idx, safe_gid]。", "改为静态迭代后着色暴降 4.4x（286.5 -> 64.9 ms），单芯直冲 12,843 FPS、4 芯 50,131 FPS（gather_ops = 0）。"],
          ["负结论反演", "3DGS 初版 Pallas 负优化", "2. 手写 Pallas 3DGS 反输给纯 XLA 1.81 倍", "初版 MXU 对数空间 Pallas 核（gs3d_pallas）仅跑 658.6 FPS，大幅落后纯 XLA jax_tile（1,193.2 FPS）——原因是 (BT, 64, 128) 对齐填充让写出膨胀 16 倍且触发 65,536 次碎 DMA。", "先如实保留负结论与 jax_tile 基线，随后重做 VMEM 分块 EWA 排序（gs3d_v3），实现 119 倍逆袭（4 芯 14.2 万 / 32 芯 95.2 万 FPS、10.62 亿高斯）。"],
          ["3 连败 -> 全过门", "神经着色器门禁消融", "3. Stage-2 神经着色器前 3 轮消融全败", "纯像素级 MLP 无感受野导致软阴影发虚（27.80 dB）；窄 W32 U-Net 夹爪遮挡错位 10%；Loss 内加 jnp.clip 直接在 Step-1 杀死梯度（卡死 15.20 dB）。", "加入 reflect(V, N) + Fresnel N·V、双 1x1 BRDF 跳连、去 clip 全局梯度裁剪并补强紧握样本后，38.35 dB / 0.9889 SSIM 8/8 项门禁全过。"],
          ["多机踩坑 -> 已实测打通", "32 芯整集群 395.1 s", "4. 跨节点 MJX 非本地数组与多行 SSH 清理残留", "8 机（32 芯）分布式启动时，mjx.put_model(m, impl='jax') 默认 device=None 会生成跨机数组并触发 IFRT 崩溃；早期多行 PID 换行清理曾锁住 /dev/vfio/3。", "修补 device=local_devs[0] 与单行 tr 清理后，在 8 机独占窗口内仅用 395.07 s 跑通全部 32 芯压测：41.1 万 FPS E2E、95.2 万 FPS 3DGS、10.62 亿高斯、638.8 亿 LiDAR 射线/秒、171.8 亿唯一面与 8K 1,650 spm，15 分钟内清回 8/8 IDLE 归还。"]
        ],
        ablationRows: [
          ["Phase-0 锁定硬阈值", ">= 30.0 dB", ">= 28.0 dB", "<= 0.010", "<span class='badge cyan'>必须 8/8 全通过</span>"],
          ["Stage 1 解析直接光基线", "13.83 dB", "—", "—", "仅硬阴影，无软阴影与间接光"],
          ["M1 像素级 MLP (1x1, 150 ep)", "27.80 dB", "23.52 dB", "0.0500 (7/140)", "<span class='badge fail'>FAIL</span> 缺少空间感受野"],
          ["M2 空间 U-Net (窄通道 W32)", "35.58 dB", "27.79 dB", "0.1000 (14/140)", "<span class='badge fail'>FAIL</span> 接触区域边缘错位"],
          ["M2 空间 U-Net + jnp.clip Loss", "15.20 dB", "—", "—", "<span class='badge fail'>FAIL</span> Step-1 梯度饱和归零"],
          ["M2 空间 U-Net (24-ch BRDF 终版)", "<strong>38.35 dB</strong>", "<strong>30.31 dB</strong>", "<strong>0.00714 (1/140)</strong>", "<span class='badge ok'>PASS (8/8)</span> SSIM 0.9889, 边缘 0.157 px"]
        ],
        rejected: [
          {
            id: "rejected_openusd_clay",
            img: "./assets/postmortem/rejected_openusd_clay.webp",
            banner: "QA 已拦截淘汰 · 早期未绑定贴图的“白模泥塑感”",
            title: "淘汰样本 A：早期 OpenUSD 几何管线直出（v3_w2_openusd_4k）",
            desc: "问题诊断：在完整接入迪士尼 Ptex/材质节点编译之前，远景植被与水面反射退化成平淡白模。已由第 03 章的晴天泻湖、黄金日落与 Moana×MJX 4K 成片全面替代。"
          },
          {
            id: "rejected_garden_seams",
            img: "./assets/postmortem/rejected_garden_seams.webp",
            banner: "QA 已拦截淘汰 · 明显 16x16 TILE 拼缝与黑底裁切",
            title: "淘汰样本 B：早期 32 视角 3DGS 训练可视化（v3_w3_garden_32v_grad）",
            desc: "问题诊断：早期 tile 局部累积边界直接截断了跨 16x16 tile 的高斯尾迹，导致画面出现肉眼可见的方格拼缝与黑底裁切。成片已淘汰并公开展示反面样本，对应 31.31 dB 数值训练日志已单独验证真实。"
          }
        ],
        honest: "全页所有指标——涵盖单芯、单机 4 芯（383 GiB HBM）以及完整 8 机 32 芯 TPU v5p-64 集群（3.064 TB HBM，v3_pod_v5p64_capstone.json）——均为真机实测结果，零注水外推。尚未单独上传 JSON 跑分的标准套件（RoboDojo、MuJoCo Playground、ALOHA、ManiSkill）均明确标注为「接口直连兼容」。同时我们坦诚记录 GPU 依然绝对领先的领域：亚秒级互动编辑器响应、硬件 RT Core 应对随机多反弹路径分支，以及单卡 Cycles 渲染速度（Chai et al., 2026）。",
        authorBio: "聚焦编译器原生图形渲染、可微物理仿真与 TPU/AI 加速器高吞吐具身智能系统的研究者与工程师。",
        ack: "第三方资产与参考致谢：Walt Disney Animation Studios Moana Island Scene v2.1（非商业研究许可）；Blender Open Data / Blender Studio 场景（BMW27、Barcelona Pavilion、Barbershop、Classroom、Junkshop、Lone Monk、Monster Under the Bed、Fishy Cat — CC0 / CC-BY）；Mip-NeRF 360 真实摄影测量数据集；Google DeepMind mujoco_menagerie（Apache-2.0 / BSD）；Chai 等（2026）wenhaochai/rtx-h100 H100/RTX PRO 6000 Cycles 基准数据。"
      },
      podStats: [
        ["基准 1 & 2 · 具身物理 + 3DGS (4 芯 -> 32 芯)", "411,023 FPS E2E  |  952,248 FPS 3DGS", "从 4 芯 5.15 万 FPS 扩展 7.97x 至 32 芯 41.10 万 FPS（B=131k）；128² 3DGS 达 95.22 万 FPS，外加 ICI 分布式 10.62 亿高斯 @ 1080p（3.41 s）。"],
        ["基准 3 · 128x2048 批量 360° 激光雷达 (32 芯)", "243,676 env-FPS  |  每秒 638.8 亿条射线", "8 机 32 芯上 B=16,384 并发 128x2048 线 LiDAR 仅 67.2 ms（B=32,768 达 229,060 env-FPS / 600.5 亿射线/秒，gather_ops=0）。"],
        ["基准 4 · 几何分片唯一三角面上限 (32 芯)", "171.8 亿唯一非实例化三角面 (1.18 TB HBM)", "ICI jax.lax.pmin 相对 4 芯 21.5 亿实现整整 8x 扩展：171.8 亿唯一面 4K 4.46 s / 8K 8.53 s（85.9 亿面 4K 2.43 s / 8K 4.70 s）。"],
        ["基准 5 · 空间分片原生 8K 3-Bounce 路径追踪 (32 芯)", "7680x4320 达 1,650.2 spm (7.54x 扩展)", "3,318 万像素无偏 3 次反弹蒙特卡洛全局光（5.475 GRays/s）：4 spp 仅 145.4 ms（6.88 FPS），1 spp 达 37.5 ms（26.69 FPS，零回调）。"]
      ],
      compRows: [
        ["具身视觉 RL、VLA 与触觉", "<strong>单张 XLA 零拷贝图</strong>：4 芯 <code>266k FPS</code> / 32 芯 <code>411k FPS</code>、142.6s 生成百万帧、GelSight <code>270.6k/s</code>", "静态离线渲染器（无交互式 RL 闭环）", "成熟的 CUDA Warp / Isaac 生态与 RTX/H100 工具链"],
        ["Real-to-Sim 3DGS / 4DGS + 物理", "<strong>零 Gather 原生 TPU</strong>：<code>14.2 万</code> (4芯) / <code>95.2 万</code> (32芯)、<code>10.62 亿</code>分布式高斯、<code>8K garden 29.2 MPix/s</code>、实时 <code>4DGS x MJX</code>", "仅支持 Cycles Mesh + NRC", "需要独立 CUDA gsplat 与 USD/PhysX 跨端桥接"],
        ["端到端可微物理 + 光追渲染", "<strong>36.08 dB / 末端误差 1.63 cm</strong>：<code>jax.value_and_grad(mjx.step + PBR)</code> 实测 <code>27.9 grad-steps/s</code>", "不可微离线路径追踪器", "需单独组合 Warp 可微物理与 Nvdiffrast"],
        ["电影 OpenUSD 与 Blender 路径追踪", "<strong>超大场景编译上限</strong>：迪士尼 <em>Moana</em>（<code>223 亿</code>面仅 4.65 GiB）、<code>BMW27 37.65 dB</code>、32 芯 <code>171.8 亿</code>唯一面", "<strong>绝对速度标杆 (wenhaochai/rtx-h100)</strong>：单卡 H100 达 <code>55,148 spm</code>（7.8x Blender），RTX PRO 6000 OptiX 达 <code>78,297 spm</code>（NRC 39.9 img/s）", "完整 Omniverse / OptiX 实时互动视口"],
        ["AI 重打光、降噪与超分 DLSS", "<strong>原生 MXU BF16/INT8</strong>：4K 降噪 <code>87.8 FPS</code>、4K DLSS <code>58.9 FPS</code>、零光追重打光 4K <code>78.6 FPS</code> / 1080p <code>307 FPS</code>", "NRC-scatter <code>40.4 img/s</code>（H100 32-spp）", "专用 Tensor Core 硬件 DLSS 3.5 与光线重建"]
      ],
      ui: {
        copy: "复制", copied: "已复制", zoom4k: "检视原图 ↗", loupe: "1:1 像素放大镜 (L)", raw: "打开完整原图 ↗",
        footerL: "TPU-RTX · Google Cloud TPU v5p (JAX + Pallas Mosaic · 零 host_callbacks · 零 HBM Gather)",
        searchPlaceholder: "搜索实验名称、场景或 JSON 文件名...",
        v3_head: "TPU-RTX 实机渲染动态视频展厅（原生 4K / 1080p · 30 fps）",
        v5_head: "具身机器人动态步态、可微物理伺服与 Menagerie 视频展厅（30 fps）",
        vid_all: "全部视频 (11)", vid_gs3d: "4K 3DGS / 4DGS (6)", vid_optics: "4K 虚化与布料 (3)", vid_relight: "神经重打光 (2)",
        b_mjx: "具身与机器人仿真", b_gs3d: "3DGS / 4DGS / 10.6 亿集群", b_cinema: "Moana / 布料 / Blender", b_dlss: "TPU-DLSS 3.5 与重打光", b_fx: "路径追踪与物理光学",
        g_all: "全部单场景 (26)", g_film: "Moana 与 Blender Studio (10)", g_optics: "物理光学与焦散 (3)", g_gs3d: "实景 3DGS / Bistro (6)", g_robotics: "Menagerie 与 RoboTwin (5)", g_rej: "QA 淘汰案例 (2)",
        th1: "任务类型", th2: "TPU-RTX (Google TPU v5p)", th3: "GPU Cycles (wenhaochai/rtx-h100)", th4: "NVIDIA Isaac Sim / OptiX",
        gvt1: "对比维度", gvt2: "Google Cloud TPU v5p (TPU-RTX)", gvt3: "NVIDIA GPU (RTX / H100 / Isaac)",
        bs1: "Benchmark / 套件", bs2: "适配模块栈", bs3: "状态", bs4: "真机实测指标或兼容范围",
        ab1: "Stage-2 神经着色器消融阶段", ab2: "PSNR (dB)", ab3: "夹爪接触区 PSNR", ab4: "遮挡错位率", ab5: "门禁结果与归因",
        ld_all: "全部 (21)", ld_rob: "具身与仿真", ld_film: "电影与布料", ld_gs: "3DGS/4DGS", ld_nn: "神经/光学",
        lt1: "实验项目", lt2: "配置与规模", lt3: "实测吞吐", lt4: "精度与对齐状态", lt5: "JSON",
        ev_t: "时间顺序步骤包络图 · 公开负优化实验", ev_h: "587x 光追管线演进曲线（453 -> 266,209 FPS · 32 芯 411,023 FPS）", ev_s: "按时间标注每次新高及平缓/负优化实验（参考 wenhaochai/rtx-h100 图表范式）。",
        gs_t: "对数尺度反转 · 3D 高斯溅射", gs_h: "3DGS 演进曲线：负优化 (-1.81x) -> 14.2 万 (4芯) -> 95.2 万 FPS (32芯)", gs_s: "放弃膨胀写出的初版 Pallas 后重构 VMEM 分块 EWA，实现 142,008 FPS (4芯) 与 952,248 FPS + 10.62 亿高斯 (32芯)。",
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
