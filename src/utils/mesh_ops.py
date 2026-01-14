import torch
import numpy as np
import trimesh
from skimage import measure

def generate_mesh_from_sdf(decoder, features, resolution=64, threshold=0.0, device='cuda', max_batch=32768):
    """
    Reconstructs a mesh from the implicit SDF decoder using Marching Cubes.
    
    Args:
        decoder: ImplicitDecoder model
        features: (1, C) Global features for one specific object
        resolution: Grid resolution (e.g., 64 -> 64x64x64 grid)
        threshold: SDF iso-surface value (usually 0.0)
        device: 'cuda' or 'cpu'
        max_batch: Max points to query at once to avoid OOM
        
    Returns:
        mesh: trimesh.Trimesh object (or None if empty)
    """
    decoder.eval()
    
    # 1. Create a 3D grid
    # Grid range [-0.5, 0.5] centered at 0
    grid_coords = np.linspace(-0.5, 0.5, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    
    # (N^3, 3) points
    points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    points_tensor = torch.from_numpy(points).float().to(device)
    
    # Expand features for batch processing if needed, but here we process chunks
    features = features.to(device)
    
    # 2. Query SDF in chunks
    sdf_values = []
    
    with torch.no_grad():
        num_points = points.shape[0]
        for i in range(0, num_points, max_batch):
            chunk_points = points_tensor[i : i + max_batch].unsqueeze(0) # (1, N_chunk, 3)
            
            # Forward pass: inputs (1, C), (1, N_chunk, 3)
            # Output: (1, N_chunk, 1)
            chunk_sdf = decoder(features, chunk_points)
            
            sdf_values.append(chunk_sdf.squeeze(0).squeeze(-1).cpu().numpy())
            
    # Combine chunks
    sdf_values = np.concatenate(sdf_values, axis=0) # (N^3,)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    # 3. Marching Cubes
    # Check if we have both positive and negative values (surface crossing)
    if sdf_grid.min() > threshold or sdf_grid.max() < threshold:
        print(f"Warning: No surface found at level {threshold} (SDF range: {sdf_grid.min():.4f} to {sdf_grid.max():.4f})")
        return None
        
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=threshold)
        
        # Scale vertices back to [-0.5, 0.5]
        # marching_cubes returns coords in [0, resolution-1]
        verts = verts / (resolution - 1) - 0.5
        
        # Create Trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        return mesh
        
    except ValueError as e:
        print(f"Marching Cubes failed: {e}")
        return None
