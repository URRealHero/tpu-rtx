"""Apply a .blend scene's colour management (view transform, look, exposure) to a linear image.

blender -b scene.blend --python blender_apply_view.py -- linear.npy out.png
The .npy is (H, W, 3) linear Rec.709, top row first (as the twin writes it).
"""

import sys

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
lin = np.load(a[0]).astype(np.float32)
H, W = lin.shape[:2]
sc = bpy.context.scene
img = bpy.data.images.new("lin", width=W, height=H, alpha=True, float_buffer=True)
img.colorspace_settings.name = "Linear Rec.709"
rgba = np.concatenate([lin[::-1], np.ones((H, W, 1), np.float32)], axis=-1)  # Blender rows are bottom-up
img.pixels.foreach_set(rgba.ravel())
ims = sc.render.image_settings
if hasattr(ims, "media_type"):
  ims.media_type = "IMAGE"
ims.file_format = "PNG"
ims.color_depth = "8"
ims.color_mode = "RGB"
img.save_render(filepath=a[1], scene=sc)
print("APPLY_VIEW_OK", sc.view_settings.view_transform, sc.view_settings.look, sc.view_settings.exposure, a[1])
