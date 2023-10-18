import random

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from . import utils
from lib.common.obj import compute_normal


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
        try:
            self.glctx = dr.RasterizeCudaContext()
        except:
            self.glctx = dr.RasterizeGLContext()

    def forward(self, mesh, mvp,
                h=512,
                w=512,
                light_d=None,
                ambient_ratio=1.,
                shading='albedo',
                spp=1,
                mlp_texture=None,
                is_train=False):
        """
        Args:
            spp:
            return_normal:
            transform_nml:
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            ambient_ratio: float
            shading: str shading type albedo, normal,
            ssp: int
        Returns:
            color: [batch, h, w, 3]
            alpha: [batch, h, w, 1]
            depth: [batch, h, w, 1]

        """
        B = mvp.shape[0]
        v_clip = torch.bmm(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1),
                           torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]

        if is_train:
            vn, _ = compute_normal(v_clip[0, :, :3], mesh.f)
            normal, _ = dr.interpolate(vn[None, ...].float(), rast, mesh.f)
        else:
            normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)

        # Texture coordinate
        if not shading == 'normal':
            if mlp_texture is not None:
                albedo = self.get_mlp_texture(mesh, mlp_texture, rast, rast_db)
            else:
                albedo = self.get_2d_texture(mesh, rast, rast_db)

        if shading == 'normal':
            color = (normal + 1) / 2.
        elif shading == 'albedo':
            color = albedo
        else:  # lambertian
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = albedo * lambertian.repeat(1, 1, 1, 3)

        normal = (normal + 1) / 2.

        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        # inverse super-sampling
        if spp > 1:
            color = utils.scale_img_nhwc(color, (h, w))
            alpha = utils.scale_img_nhwc(alpha, (h, w))
            normal = utils.scale_img_nhwc(normal, (h, w))

        return color, normal, alpha

    def get_mlp_texture(self, mesh, mlp_texture, rast, rast_db, res=2048):
        # uv = mesh.vt[None, ...] * 2.0 - 1.0
        uv = mesh.vt[None, ...]

        # pad to four component coordinate
        uv4 = torch.cat((uv, torch.zeros_like(uv[..., 0:1]), torch.ones_like(uv[..., 0:1])), dim=-1)

        # rasterize
        _rast, _ = dr.rasterize(self.glctx, uv4, mesh.f.int(), (res, res))
        print("_rast ", _rast.shape)
        # Interpolate world space position
        # gb_pos, _ = dr.interpolate(mesh.v[None, ...], _rast, mesh.f.int())

        # Sample out textures from MLP
        tex = mlp_texture.sample(_rast[..., :-1].view(-1, 3)).view(*_rast.shape[:-1], 3)

        texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all')
        print(tex.shape)

        albedo = dr.texture(
            tex, texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
        # albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background

        # print(tex.shape, albedo.shape)
        # exit()
        return albedo

    @staticmethod
    def get_2d_texture(mesh, rast, rast_db):
        texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all')

        albedo = dr.texture(
            mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background
        return albedo
