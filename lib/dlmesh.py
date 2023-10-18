import os
import random

import numpy as np
import trimesh
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
# from apps.mp import draw_landmarks_on_image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from lib.encoding import get_encoder
from lib.common.obj import Mesh, safe_normalize, normalize_vert, save_obj_mesh, compute_normal
from lib.common.utils import trunc_rev_sigmoid, SMPLXSeg
from lib.common.renderer import Renderer
from lib.common.remesh import smplx_remesh_mask, subdivide, subdivide_inorder
from lib.common.lbs import warp_points
from lib.common.visual import draw_landmarks


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class DLMesh(nn.Module):
    def __init__(self, opt, num_layers_bg=2, hidden_dim_bg=16):

        super(DLMesh, self).__init__()

        self.opt = opt

        self.num_remeshing = 1

        self.renderer = Renderer()

        self.device = torch.device("cuda")
        self.lock_beta = opt.lock_beta

        if self.opt.lock_geo:  # texture
            self.mesh = Mesh.load_obj(self.opt.mesh)
            self.mesh.auto_normal()
        else:  # geometry
            self.body_model = smplx.create(
                model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
                model_type='smplx',
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True,
                create_left_hand_pose=True,
                create_right_hand_pose=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                create_expression=True,
                create_transl=False,
                use_pca=False,
                use_face_contour=True,
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                num_pca_comps=12,
                dtype=torch.float32,
                batch_size=1,
            ).to(self.device)

            self.smplx_faces = self.body_model.faces.astype(np.int32)

            param_file = "./data/init_body/fit_smplx_params.npz"
            smplx_params = dict(np.load(param_file))
            self.betas = torch.as_tensor(smplx_params["betas"]).to(self.device)
            self.jaw_pose = torch.as_tensor(smplx_params["jaw_pose"]).to(self.device)
            self.body_pose = torch.as_tensor(smplx_params["body_pose"]).to(self.device)
            self.body_pose = self.body_pose.view(-1, 3)
            self.body_pose[[0, 1, 3, 4, 6, 7], :2] *= 0
            self.body_pose = self.body_pose.view(1, -1)

            self.global_orient = torch.as_tensor(smplx_params["global_orient"]).to(self.device)
            self.expression = torch.zeros(1, 100).to(self.device)

            self.remesh_mask = self.get_remesh_mask()
            self.faces_list, self.dense_lbs_weights, self.uniques, self.vt, self.ft = self.get_init_body()

            N = self.dense_lbs_weights.shape[0]

        # background network
        if not self.opt.skip_bg:
            self.encoder_bg, in_dim_bg = get_encoder('frequency_torch', multires=4)
            self.bg_net = MLP(in_dim_bg, 3, hidden_dim_bg, num_layers_bg)

        self.mlp_texture = None
        if not self.opt.lock_tex:  # texture parameters
            if self.opt.tex_mlp:
                # self.encoder_tex, self.in_dim = get_encoder('hashgrid', interpolation='smoothstep')
                # self.tex_net = MLP(self.in_dim, 3, 32, 2)
                from .mlptexture import MLPTexture3D
                self.mlp_texture = MLPTexture3D()
            else:
                res = self.opt.albedo_res
                albedo = torch.ones((res, res, 3), dtype=torch.float32) * 0.5  # default color
                self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(albedo))

        # Geometry parameters
        if not self.opt.lock_geo:
            # displacement
            if self.opt.geo_mlp:
                self.encoder_geo, in_dim_geo = get_encoder('hashgrid', interpolation='smoothstep')
                self.geo_net = MLP(in_dim_geo, 1, 32, 2)
            else:
                self.v_offsets = nn.Parameter(torch.zeros(N, 1))

            # shape
            if not self.lock_beta:
                self.betas = nn.Parameter(self.betas)

            # expression
            rich_data = np.load("./data/talkshow/rich.npy")
            self.rich_params = torch.as_tensor(rich_data, dtype=torch.float32, device=self.device)
            if not self.opt.lock_expression:
                self.expression = nn.Parameter(self.expression)

            if not self.opt.lock_pose:
                self.body_pose = nn.Parameter(self.body_pose)

            self.jaw_pose = nn.Parameter(self.jaw_pose)

        # Create an FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path='data/mediapipe/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    @torch.no_grad()
    def get_init_body(self, cache_path='./data/init_body/data.npz'):
        if True:
            if self.num_remeshing == 1:
                cache_path = './data/init_body/data.npz'
                data = np.load(cache_path)
                faces_list = [torch.as_tensor(data['dense_faces'], device=self.device)]
                dense_lbs_weights = torch.as_tensor(data['dense_lbs_weights'], device=self.device)
                unique_list = [data['unique']]
                vt = torch.as_tensor(data['vt'], device=self.device)
                ft = torch.as_tensor(data['ft'], device=self.device)
            else:
                cache_path = './data/init_body/data-remesh2.npz'
                data = np.load(cache_path, allow_pickle=True)
                faces_list = [torch.as_tensor(f, device=self.device) for f in data["faces"]]
                dense_lbs_weights = torch.as_tensor(data['dense_lbs_weights'], device=self.device)
                unique_list = data['uniques']
                vt = torch.as_tensor(data['vt'], device=self.device)
                ft = torch.as_tensor(data['ft'], device=self.device)
        else:
            output = self.body_model(
                betas=self.betas,
                body_pose=self.body_pose,
                jaw_pose=self.jaw_pose,
                expression=self.expression,
                return_verts=True
            )
            v_cano = output.v_posed[0]

            # re-meshing
            dense_v_cano, dense_faces, dense_lbs_weights, unique = subdivide(v_cano.cpu().numpy(),
                                                                             self.smplx_faces[self.remesh_mask],
                                                                             self.body_model.lbs_weights.detach().cpu().numpy())
            dense_faces = np.concatenate([dense_faces, self.smplx_faces[~self.remesh_mask]])

            unique_list = [unique]
            faces_list = [dense_faces]
            # re-meshing
            for _ in range(1, self.num_remeshing):
                dense_v_cano, dense_faces, dense_lbs_weights, unique = subdivide(dense_v_cano, dense_faces,
                                                                                 dense_lbs_weights)
                unique_list.append(unique)
                faces_list.append(dense_faces)

            dense_v = torch.as_tensor(dense_v_cano, device=self.device)
            dense_faces = torch.as_tensor(dense_faces, device=self.device)
            dense_lbs_weights = torch.as_tensor(dense_lbs_weights, device=self.device)

            dense_v_posed = warp_points(dense_v, dense_lbs_weights, output.joints_transform[:, :55])[0]

            dense_v_posed = normalize_vert(dense_v_posed)

            vt, ft = Mesh(device=self.device).auto_uv(v=dense_v_posed, f=dense_faces)

            np.savez(
                cache_path,
                faces=np.array(faces_list, dtype=object),
                dense_lbs_weights=dense_lbs_weights.cpu().numpy(),
                uniques=np.array(unique_list, dtype=object),
                vt=vt.cpu().numpy(),
                ft=ft.cpu().numpy()
            )

            # trimesh.Trimesh(dense_v_posed.cpu().numpy(), dense_faces.cpu().numpy()).export("mesh.obj")
            # exit()

            # exit()

        return faces_list, dense_lbs_weights, unique_list, vt, ft

    def get_remesh_mask(self):
        ids = list(set(SMPLXSeg.front_face_ids) - set(SMPLXSeg.forehead_ids))
        ids = ids + SMPLXSeg.ears_ids + SMPLXSeg.eyeball_ids + SMPLXSeg.hands_ids
        mask = ~np.isin(np.arange(10475), ids)
        mask = mask[self.body_model.faces].all(axis=1)
        return mask

    def get_params(self, lr):
        params = []

        if not self.opt.skip_bg:
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        if not self.opt.lock_tex:
            if self.opt.tex_mlp:
                params.extend([
                    {'params': self.mlp_texture.parameters(), 'lr': lr * 10},
                ])
            else:
                params.append({'params': self.raw_albedo, 'lr': lr * 10})

        if not self.opt.lock_geo:
            if self.opt.geo_mlp:
                params.extend([
                    {'params': self.encoder_geo.parameters(), 'lr': lr * 10},
                    {'params': self.geo_net.parameters(), 'lr': lr},
                ])
            else:
                params.append({'params': self.v_offsets, 'lr': 0.0001})

            if not self.lock_beta:
                params.append({'params': self.betas, 'lr': 0.1})

            if not self.opt.lock_expression:
                params.append({'params': self.expression, 'lr': 0.05})

            if not self.opt.lock_pose:
                params.append({'params': self.body_pose, 'lr': 0.05})
                params.append({'params': self.jaw_pose, 'lr': 0.05})

        return params

    def get_vertex_offset(self, is_train):
        v_offsets = self.v_offsets
        if not is_train and self.opt.replace_hands_eyes:
            v_offsets[SMPLXSeg.eyeball_ids] = 0.
            v_offsets[SMPLXSeg.hands_ids] = 0.
        return v_offsets

    def get_mesh(self, is_train):
        # os.makedirs("./results/pipline/obj/", exist_ok=True)
        if not self.opt.lock_geo:
            output = self.body_model(
                betas=self.betas,
                body_pose=self.body_pose,
                jaw_pose=self.jaw_pose,
                # jaw_pose=random.choice(self.rich_params)[None, :3],
                # jaw_pose=self.rich_params[500:501, :3],
                expression=self.expression,
                return_verts=True
            )
            v_cano = output.v_posed[0]
            landmarks = output.joints[0, -68:, :]

            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_faces[self.remesh_mask], self.uniques[0])

            for unique, faces in zip(self.uniques[1:], self.faces_list[:-1]):
                v_cano_dense = subdivide_inorder(v_cano_dense, faces, unique)

            # add offset before warp
            if not self.opt.lock_geo:
                if self.v_offsets.shape[1] ==1:
                    vn = compute_normal(v_cano_dense, self.faces_list[-1])[0]
                    v_cano_dense += self.get_vertex_offset(is_train) * vn
                else:
                    v_cano_dense += self.get_vertex_offset(is_train)
            # LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights,
                                        output.joints_transform[:, :55]).squeeze(0)

            # if not is_train:
            v_posed_dense, center, scale = normalize_vert(v_posed_dense, return_cs=True)

            mesh = Mesh(v_posed_dense, self.faces_list[-1].int(), vt=self.vt, ft=self.ft)
            mesh.auto_normal()

            if not self.opt.lock_tex and not self.opt.tex_mlp:
                mesh.set_albedo(self.raw_albedo)

        else:
            mesh = Mesh(base=self.mesh)
            mesh.set_albedo(self.raw_albedo)
        return mesh, landmarks

    @torch.no_grad()
    def get_mesh_center_scale(self, phrase):
        assert phrase in ["face", "body"]
        vertices = self.body_model(
            betas=self.betas,
            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True).vertices[0]
        vertices = normalize_vert(vertices)

        if phrase == "face":
            vertices = vertices[SMPLXSeg.head_ids + SMPLXSeg.neck_ids]
        max_v = vertices.max(0)[0]
        min_v = vertices.min(0)[0]
        scale = (max_v[1] - min_v[1])
        center = (max_v + min_v) * 0.5
        # center = torch.mean(points, dim=0, keepdim=True)
        return center, scale

    @torch.no_grad()
    def export_mesh(self, save_dir):
        mesh = self.get_mesh(is_train=False)[0]
        obj_path = os.path.join(save_dir, 'mesh.obj')
        mesh.write(obj_path)

    @torch.no_grad()
    def get_mediapipe_landmarks(self, image):
        """
        Parameters
        ----------
        image: np.ndarray HxWxC

        Returns
        -------
        face_landmarks_list
        """
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.astype(np.uint8))
        detection_result = self.detector.detect(image_mp)
        face_landmarks_list = detection_result.face_landmarks
        return face_landmarks_list

    def forward(self, rays_o, rays_d, mvp, h, w, light_d=None, ambient_ratio=1.0, shading='albedo', is_train=True):

        batch = rays_o.shape[0]

        if not self.opt.skip_bg:
            dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            bg_color = torch.sigmoid(self.bg_net(self.encoder_bg(dirs.view(-1, 3)))).view(batch, h, w, 3).contiguous()
        else:
            bg_color = torch.ones(batch, h, w, 3).to(mvp.device)

        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=rays_o.device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        # render
        pr_mesh, smplx_landmarks = self.get_mesh(is_train=is_train)

        rgb, normal, alpha = self.renderer(pr_mesh, mvp, h, w, light_d, ambient_ratio, shading, self.opt.ssaa,
                                           mlp_texture=self.mlp_texture, is_train=is_train)
        rgb = rgb * alpha + (1 - alpha) * bg_color
        normal = normal * alpha + (1 - alpha) * bg_color

        # smplx landmarks
        smplx_landmarks = torch.bmm(F.pad(smplx_landmarks, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0),
                                    torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]
        smplx_landmarks = smplx_landmarks[..., :2] / smplx_landmarks[..., 2:3]
        smplx_landmarks = smplx_landmarks * 0.5 + 0.5

        return {
            "image": rgb,
            "alpha": alpha,
            "normal": normal,
            "smplx_landmarks": smplx_landmarks,
        }
