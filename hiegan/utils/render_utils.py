# hiegan/utils/render_utils.py
import torch
from typing import Optional
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, TexturesVertex, FoVPerspectiveCameras
)

def simple_renderer(image_size: int = 256, device: str = "cuda") -> MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    cameras = FoVPerspectiveCameras(device=device)
    lights = PointLights(device=device)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader
    )

@torch.no_grad()
def render_mesh_rgb(mesh, renderer: Optional[MeshRenderer] = None, device: str = "cuda"):
    """
    Quick RGB render. If mesh has no textures, add a per-vertex gray color.
    Returns image tensor: (B, H, W, 3), values in [0,1]
    """
    if renderer is None:
        renderer = simple_renderer(device=device)

    # Ensure textures exist
    if mesh.textures is None:
        verts = mesh.verts_list()[0]
        colors = torch.ones_like(verts) * 0.7
        mesh.textures = TexturesVertex(verts_features=[colors.to(verts.device)])

    images = renderer(mesh)
    return images.clamp(0, 1)
