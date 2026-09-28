"""
Pure-PyTorch differentiable orthographic Gaussian rasterizer.

This is a lightweight, self-contained stand-in for StreamSplat's CUDA
``diff-gaussian-rasterization-orth`` rasterizer. It reproduces the *interface*
and the *physics* of the real one (orthographic projection, alpha-composited
splatting, per-pixel depth) so the event-driven dynamic decoder can be
developed and demonstrated without building CUDA kernels. The real CUDA
rasterizer in ``gaussian_renderer_dynamic`` is a drop-in replacement:

    from gaussian_renderer_dynamic import render   # CUDA, used in production
    from model.torch_rasterizer import render_orth # pure torch, used here

The forward pass is fully differentiable w.r.t. means3D, colors, opacities,
scales and rotations, which is what the event-consistency loss needs in order
to backprop the event-generation residual onto the Gaussian increments.
"""
import math
import torch
import torch.nn.functional as F


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert (N,4) unit quaternions (w,x,y,z) to (N,3,3) rotation matrices."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def render_orth(
    means3D: torch.Tensor,      # [N, 3]  (x, depth, z) in normalized ortho volume
    colors: torch.Tensor,       # [N, 3]  precomputed RGB
    opacities: torch.Tensor,    # [N, 1]
    scales: torch.Tensor,       # [N, 3]  per-axis scale (std-dev of the Gaussian)
    rotations: torch.Tensor,    # [N, 4]  quaternion
    H: int,
    W: int,
    bg: torch.Tensor = None,    # [3]
    *,
    chunk: int = 1 << 16,
):
    """Orthographic alpha-composited splatting.

    The ortho camera maps world (x, z) directly to image (u, v) and uses the
    second coordinate (y) as depth. This mirrors StreamSplat's fixed camera
    ``R_fixed`` (see ``gaussian_renderer_dynamic/__init__.py``), which swaps
    axes so that the camera looks along +y and the image plane spans (x, z).
    Output resolution maps the normalized volume [-1, 1] -> [0, H) x [0, W).
    """
    device = means3D.device
    N = means3D.shape[0]
    if bg is None:
        bg = torch.full((3,), 0.5, device=device)

    # Map normalized volume to pixel coords. x->u (width), z->v (height).
    # In StreamSplat h_coords go 1..-1 (top to bottom), matching z below.
    u = (means3D[:, 0] * 0.5 + 0.5) * (W - 1)          # [N]
    v = (1.0 - (means3D[:, 2] * 0.5 + 0.5)) * (H - 1)  # [N], flip z so +z is up
    depth = means3D[:, 1]                                # [N]

    R = quaternion_to_matrix(rotations)  # [N,3,3]
    S = torch.diag_embed(scales.clamp(min=1e-6))  # [N,3,3]
    RS = R @ S
    cov = RS @ RS.transpose(1, 2)  # [N,3,3] covariance

    # 2D image-plane covariance: project onto (u, v) using the (x, z) block.
    # Order: index 0 -> u (x), index 2 -> v (z).
    idx = torch.tensor([0, 2], device=device)
    cov2d = cov[:, idx][:, :, idx]  # [N,2,2]
    # The splat footprint in pixels scales the volume->pixel mapping by W/2, H/2.
    pix_scale = torch.tensor([[(W - 1) / 2.0, 0.0], [0.0, (H - 1) / 2.0]],
                             device=device)
    cov2d = pix_scale @ cov2d @ pix_scale  # [N,2,2] in pixel units

    # Det-determinant and inverse of 2x2 covariance.
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    d = cov2d[:, 1, 1]
    det = a * d - b * b
    det = det.clamp(min=1e-8)
    inv_a = d / det
    inv_b = -b / det
    inv_d = a / det

    # Pixel grid.
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [HW, 2]

    color = torch.zeros(H * W, 3, device=device)
    depth_out = torch.zeros(H * W, device=device)
    alpha_out = torch.zeros(H * W, device=device)

    # Tile Gaussians in chunks to bound peak memory (O(N*HW) otherwise).
    # We accumulate per-Gaussian blend contributions into running totals and
    # do the background composite once at the end (alpha_out holds total alpha).
    color_acc = torch.zeros(H * W, 3, device=device)
    depth_acc = torch.zeros(H * W, device=device)
    w_acc = torch.zeros(H * W, 1, device=device)
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        mu = torch.stack([u[sl], v[sl]], dim=-1)               # [n,2]
        diff = pix[:, None, :] - mu[None, :, :]                # [HW, n, 2]
        # Mahalanobis distance: d = diff^T inv(cov) diff
        q = (inv_a[sl][None, :] * diff[..., 0] ** 2
             + 2.0 * inv_b[sl][None, :] * diff[..., 0] * diff[..., 1]
             + inv_d[sl][None, :] * diff[..., 1] ** 2)          # [HW, n]
        # 2D Gaussian evaluation. Clip exponent to avoid fp overflow for far pixels.
        gauss = torch.exp(-0.5 * q.clamp(max=40.0))             # [HW, n]
        alpha_i = gauss * opacities[sl][None, :, 0].clamp(0.0, 1.0)  # [HW, n]

        # Front-to-back alpha compositing, sorted by depth.
        order = torch.argsort(depth[sl])                        # [n]
        alpha_i = alpha_i[:, order]
        col_i = colors[sl][order]                               # [n,3]
        depth_i = depth[sl][order].unsqueeze(0).expand(H * W, -1)  # [HW, n]

        T = torch.cumprod(1.0 - alpha_i, dim=1)                 # [HW, n]
        T_prev = torch.cat([torch.ones(H * W, 1, device=device), T[:, :-1]], dim=1)
        w = alpha_i * T_prev                                    # [HW, n] blend weights
        color_acc = color_acc + (w.unsqueeze(-1) * col_i.unsqueeze(0)).sum(dim=1)
        depth_acc = depth_acc + (w * depth_i).sum(dim=1)
        w_acc = w_acc + w.sum(dim=1, keepdim=True)

    # Composite background where alpha < 1.
    a_total = w_acc.clamp(max=1.0)                              # [HW,1]
    color = color_acc + (1.0 - a_total) * bg
    color = color.view(H, W, 3).permute(2, 0, 1)               # [3,H,W]
    depth_out = depth_acc.view(H, W)                            # [H,W]
    alpha_out = a_total.view(H, W)                              # [H,W]
    return color, depth_out, alpha_out
