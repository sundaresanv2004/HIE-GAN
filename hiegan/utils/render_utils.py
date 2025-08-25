import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from .mesh_utils import mesh_to_pointcloud


def simple_renderer(image_size=256, device="cpu"):
    """Create a simple PyTorch3D renderer"""
    R, T = look_at_view_transform(dist=2.7, elev=30, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return renderer


@torch.no_grad()
def render_mesh_rgb(mesh, renderer=None, device="cpu"):
    """Renders a mesh with automatic texture handling"""
    if renderer is None:
        renderer = simple_renderer(device=device)

    # Ensure the mesh is on the correct device
    mesh = mesh.to(device)

    # Add white texture if mesh doesn't have one
    if mesh.textures is None:
        verts = mesh.verts_list()[0]
        # Create white texture for all vertices
        textures = TexturesVertex(verts_features=torch.ones_like(verts).unsqueeze(0))
        faces = mesh.faces_list()[0]
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    # Render the mesh
    images = renderer(mesh)
    # Return images without the alpha channel
    return images[..., :3].clamp(0, 1)


def denormalize_image(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize image tensor for display"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()


def plot_pointcloud_3d(ax, points, title="Point Cloud", color_by_z=True):
    """Plot 3D point cloud on given axis"""
    if points.shape[0] > 2000:  # Sample for performance
        indices = np.random.choice(points.shape[0], 2000, replace=False)
        points = points[indices]

    if color_by_z:
        colors = points[:, 2]  # Color by Z coordinate
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=colors, cmap='viridis', s=0.5, alpha=0.8)
    else:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             s=0.5, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Make axes equal
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_complete_sample(images, meshes, metadata, sample_idx=0, device="cpu"):
    """Complete visualization of image, mesh, and point cloud"""

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))

    # Extract single sample
    if len(images.shape) == 4:  # Batch of images
        img_tensor = images[sample_idx]
    else:
        img_tensor = images

    if len(img_tensor.shape) == 4:  # Multi-view
        img_tensor = img_tensor[0]  # Take first view

    mesh = meshes[sample_idx] if len(meshes) > 1 else meshes
    sample_meta = metadata[sample_idx] if isinstance(metadata, list) else metadata

    # 1. Show Input Image
    ax1 = plt.subplot(1, 3, 1)
    img_display = denormalize_image(img_tensor)
    ax1.imshow(img_display)
    ax1.set_title(f'Input Image\n{sample_meta["class_id"]}/{sample_meta["obj_id"]}')
    ax1.axis('off')

    # 2. Show Rendered Mesh
    ax2 = plt.subplot(1, 3, 2)
    try:
        renderer = simple_renderer(device=device)
        rendered_img = render_mesh_rgb(mesh, renderer, device=device)
        ax2.imshow(rendered_img[0].cpu().numpy())
        ax2.set_title(f'3D Mesh Render\nVerts: {mesh.verts_list()[0].shape[0]}')
        ax2.axis('off')
    except Exception as e:
        print(f"Mesh rendering failed: {e}")
        # Fallback: show mesh vertices as scatter
        vertices = mesh.verts_list()[0].cpu().numpy()
        ax2 = plt.subplot(1, 3, 2, projection='3d')
        if len(vertices) > 1000:
            indices = np.random.choice(len(vertices), 1000, replace=False)
            vertices = vertices[indices]
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5)
        ax2.set_title('Mesh Vertices (Fallback)')

    # 3. Show Point Cloud
    ax3 = plt.subplot(1, 3, 3, projection='3d')
    try:
        points = mesh_to_pointcloud(mesh, num_samples=2048)
        points_np = points[0].cpu().numpy()
        plot_pointcloud_3d(ax3, points_np, 'Sampled Point Cloud')
    except Exception as e:
        print(f"Point cloud generation failed: {e}")
        # Fallback: use mesh vertices
        vertices = mesh.verts_list()[0].cpu().numpy()
        plot_pointcloud_3d(ax3, vertices, 'Mesh Vertices')

    plt.tight_layout()
    plt.show()

    # Print detailed information
    print(f"\n--- Sample Details ---")
    print(f"Class: {sample_meta['class_id']}")
    print(f"Object: {sample_meta['obj_id']}")
    print(f"Image shape: {img_tensor.shape}")
    print(f"Mesh vertices: {mesh.verts_list()[0].shape[0]}")
    print(f"Mesh faces: {mesh.faces_list()[0].shape[0]}")
    print(f"Number of views: {sample_meta.get('num_views', 'Unknown')}")


def quick_batch_overview(images, meshes, metadata, max_samples=4):
    """Show overview of multiple samples in a batch"""
    num_samples = min(len(metadata), max_samples)

    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        # Top row: Images
        img_tensor = images[i] if len(images.shape) == 4 else images
        if len(img_tensor.shape) == 4:  # Multi-view
            img_tensor = img_tensor[0]

        img_display = denormalize_image(img_tensor)
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'{metadata[i]["class_id"][:8]}...')
        axes[0, i].axis('off')

        # Bottom row: Point clouds
        mesh = meshes[i] if len(meshes) > 1 else meshes
        try:
            points = mesh_to_pointcloud(mesh, num_samples=1024)
            points_np = points[0].cpu().numpy()

            axes[1, i] = plt.subplot(2, num_samples, num_samples + i + 1, projection='3d')
            plot_pointcloud_3d(axes[1, i], points_np, f'PC {i + 1}')
        except:
            vertices = mesh.verts_list()[0].cpu().numpy()
            axes[1, i] = plt.subplot(2, num_samples, num_samples + i + 1, projection='3d')
            plot_pointcloud_3d(axes[1, i], vertices, f'Verts {i + 1}')

    plt.tight_layout()
    plt.show()
