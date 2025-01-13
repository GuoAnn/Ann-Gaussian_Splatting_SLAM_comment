#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
'''这个函数的主要功能是将3D场景中的点云数据通过高斯光栅化技术渲染到2D图像上。
这种方法在计算机图形学中用于实现高质量的渲染效果，特别是在需要处理复杂光照和材质的情况下。
通过使用高斯模型，可以有效地表示和渲染场景中的每个点，从而获得更加平滑和逼真的视觉效果。'''

'''导入必要的库：
math 用于数学计算。
torch 用于张量计算。
GaussianRasterizationSettings 和 GaussianRasterizer 用于高斯光栅化设置和执行。
GaussianModel 用于处理高斯模型。
eval_sh 用于评估球谐函数'''
import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh

'''函数定义 render：
接受多个参数，包括视点相机设置、高斯模型、渲染管道设置、背景颜色、缩放修正因子、覆盖颜色和掩码。'''
def render(
    viewpoint_camera, #数据输入的所有信息
    pc: GaussianModel, #高斯模型
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 检查是否有点云数据,如果点云数据为空，则返回 None
    if pc.get_xyz.shape[0] == 0:
        return None

    #  创建一个与输入点云相同大小的零张量,用于存储屏幕空间中的点。
    screenspace_points = (
        torch.zeros_like( #创建一个与指定张量相同大小的零张量。
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad() #保留张量的梯度信息。在 PyTorch 中，当需要对张量进行梯度计算时，通常会使用 retain_grad() 方法来确保在反向传播过程中梯度信息不会丢失。
    except Exception:
        pass

    # Set up rasterization configuration（设置视场角）
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 创建光栅化设置,根据相机的视场角（FoV）计算切线值,创建 GaussianRasterizationSettings 实例，配置光栅化的各种参数。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    
    # 基于光栅化的设置来创建光栅化器,使用配置好的设置创建 GaussianRasterizer 实例。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz #获取点云数据的三维坐标，存储在 means3D 中。
    means2D = screenspace_points #获取屏幕坐标，存储在 means2D 中（也就是光栅化后2D的坐标）。
    opacity = pc.get_opacity #获取不透明度

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    #如果提供了预计算的3D协方差，则使用它；否则，根据缩放和旋转计算。
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #如果提供了预计算的颜色，则使用它；否则，根据球谐函数（SH）转换为RGB颜色。
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #光栅化：用光栅化器将可见的高斯分布渲染到图像上，并获取它们的半径（在屏幕上）。
    if mask is not None:
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    #返回一个字典，包含渲染的图像、屏幕空间点、可见性过滤器、半径、深度、不透明度和触摸计数
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }
