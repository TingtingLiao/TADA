import os
import cv2
import glob
import json
from tqdm import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.common.renderer import Renderer
from lib.common.utils import get_rays, safe_normalize
from PIL import Image
from torchvision import transforms


def _get_view_direction(thetas, phis, overhead, front):
    """
    We only encoder ['front', 'side', 'back', "overhead"] and skip "bottom"
    Args:
        thetas:
        phis:
        overhead:
        front:

    Returns:

    """
    #                   phis [B,];          thetas: [B,]
    # front = 0         [-half_front, half_front)
    # side (left) = 1   [half_front, 180 - half_front)
    # back = 2          [180 - half_front, 180+half_front)
    # side (right) = 1  [180+half_front, 360-half_front)
    # top = 3           [0, overhead]
    # bottom = 4        [180-overhead, 180]

    half_front = front / 2.
    phis_abs = phis.abs()
    res = torch.ones(thetas.shape[0], dtype=torch.long)
    res[(phis_abs <= half_front)] = 0
    # res[(phis_abs > half_front) & (phis_abs < np.pi - half_front)] = 1
    res[(phis_abs > np.pi - half_front) & (phis_abs <= np.pi)] = 2
    # override by thetas
    # res[thetas <= overhead] = 3
    # res[thetas >= (np.pi - overhead)] = 4
    return res


def get_view_direction(thetas, phis, overhead, front):
    """
    We only encoder ['front', 'side', 'back', "overhead"] and skip "bottom"
    Args:
        thetas:
        phis:
        overhead:
        front:

    Returns:

    """
    #                   phis [B,];          thetas: [B,]
    # front = 0         [-half_front, half_front)
    # side (left) = 1   [half_front, 180 - half_front)
    # back = 2          [180 - half_front, 180+half_front)
    # side (right) = 1  [180+half_front, 360-half_front)
    # top = 3           [0, overhead]
    # bottom = 4        [180-overhead, 180]

    half_front = front / 2.
    phis_abs = phis.abs()
    res = torch.ones(thetas.shape[0], dtype=torch.long)
    res[(phis_abs <= half_front)] = 0
    # res[(phis_abs > half_front) & (phis_abs < np.pi - half_front)] = 1
    res[(phis_abs > np.pi - half_front) & (phis_abs <= np.pi)] = 2
    # override by thetas
    # res[thetas <= overhead] = 3
    # res[thetas >= (np.pi - overhead)] = 4
    return res


def rand_poses(size,
               device,
               radius_range=[1.2, 1.4],
               theta_range=[80, 100],
               phi_range=[-60, 60],
               return_dirs=False,
               angle_overhead=30,
               angle_front=60,
               jitter=False,
               uniform_sphere_rate=0.5):
    """
    generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_range: [min, max] camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
        return_dirs: bool return camera direction if true
        angle_overhead: float
        angle_front: float
        jitter: bool
        uniform_sphere_rate: float should be in [0, 1]
    Return:
        poses: [size, 4, 4]
    """
    # This only works for Geometry
    # if phi_range[1] <= 45:
    #     p = random.random()
    #     if p < 0.35:  # front
    #         phi_range = np.array(phi_range)
    #     elif p < 0.5:  # side
    #         phi_range = 90 + np.array(phi_range)
    #     elif p < 0.85:  # backs
    #         if random.random() > 0.5:
    #             phi_range = [phi_range[0] + 180, 180]
    #         else:
    #             phi_range = [-180, phi_range[1] - 180]
    #     else:  # side
    #         phi_range = -90 + np.array(phi_range)

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ],
                dim=-1),
            p=2,
            dim=1)
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius


def near_head_poses(size,
                    device,
                    shift,
                    radius_range=[0.15, 0.2],
                    theta_range=[70, 90],
                    phi_range=[-60, 60],
                    return_dirs=False,
                    angle_overhead=30,
                    angle_front=60,
                    jitter=False,
                    face_scale=1.0):
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)

    # face_center_jitter = face_center + (random.random() - 0.5) * face_scale * 0.2
    # shift = torch.as_tensor([0, face_center, 0], device=device).view(1, 3)

    radius_range = np.array(radius_range) * face_scale
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) + shift  # [B, 3]
    targets = torch.zeros_like(centers) + shift

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60,
                 head_shift=[0, 0, 0]):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    shift = torch.as_tensor(head_shift, device=device).view(1, 3)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) + shift  # [B, 3]
    targets = torch.zeros_like(centers) + shift

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class ViewDataset(torch.utils.data.Dataset):

    def __init__(self, opt, device, type='train', size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        if type == 'train':
            self.H = opt.h
            self.W = opt.w
        else:
            self.H = opt.H
            self.W = opt.W
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.near
        self.far = self.opt.far  # infinite

        self.aspect = self.W / self.H
        self.full_body = False

        self.face_center = torch.as_tensor([0, 0.42, 0], device=device).view(1, 3)
        self.face_scale = 1.

        self.body_center = torch.zeros(1, 3, device=device)
        self.body_scale = 1.

        self.id_dir_map = ['front', 'side', 'back']

        self.to_tensor = transforms.Compose([
            transforms.Resize((self.W, self.H)),
            transforms.ToTensor()
        ])

        # [debug] visualize poses
        # self.test_camera()

    @staticmethod
    def modify_commandline_options(parser, full_body, face_center, face_scale, body_center, body_scale):
        return parser

    def test_camera(self):
        from lib.common.obj import Mesh
        from lib.common.utils import plot_grid_images
        mesh = Mesh.load_obj("../data/dreammesh/Apose/superman/mesh/mesh.obj", init_empty_tex=True)

        render = Renderer()

        mvp = []
        dir = []
        for i in range(100):
            data = self.__getitem__(i)
            mvp.append(data['mvp'])
            dir.append(data['dir'])
        mvp = torch.stack(mvp)
        dir = torch.stack(dir)

        rgb, normals, alpha = render(mesh, mvp, 512, 512, None, 1, "albedo")
        normals = normals.cpu().numpy()
        views = ['front', 'side', 'back', 'overhead', 'front', 'side']

        for i, d in enumerate(dir):
            cv2.putText(normals[i], views[d], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2,
                        color=(255, 255, 255),
                        thickness=6)

        plot_grid_images(normals, 10, 10, "person.png")
        print("Done!")
        exit()

    def load_images(self, path):
        rgba = Image.open(path).convert('RGBA')
        mask = rgba.split()[-1]
        rgb = rgba.convert('RGB')
        rgb = self.to_tensor(rgb)
        mask = self.to_tensor(mask)
        rgb = rgb * mask + (1 - mask)

        depth = Image.open(path.replace('_rgba.png', '_depth.png'))
        depth = self.to_tensor(depth)

        normal = Image.open(path.replace('_rgba.png', '_normal.png'))
        normal = 1 - self.to_tensor(normal)

        return rgb, mask, depth, normal[[2, 1, 0]]

    def get_default_view_data(self):
        # opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
        ref_radii, ref_polars, ref_azimuths = [], [], []
        if os.path.exists(self.opt.image):
            ref_radii = [self.opt.default_radius]
            ref_polars = [self.opt.default_polar]
            ref_azimuths = [self.opt.default_azimuth]
        else:
            # todo for multi-images
            return None

        H = int(1 * self.H)
        W = int(1 * self.W)
        cx = H / 2
        cy = W / 2

        poses, dirs = circle_poses(
            self.device,
            radius=self.opt.default_radius,
            theta=self.opt.default_polar,
            phi=self.opt.default_azimuth,
            return_dirs=True,
            angle_overhead=self.opt.angle_overhead,
            angle_front=self.opt.angle_front)

        fov = self.opt.default_fovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        projection = torch.tensor([
            [2 * focal / W, 0, 0, 0],
            [0, -2 * focal / H, 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near),
             -(2 * self.far * self.near) / (self.far - self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device)

        mvp = projection @ torch.inverse(poses.squeeze(0))  # [4, 4]

        # sample a low-resolution but full image
        rays_o, rays_d = get_rays(poses, intrinsics, self.H, self.W, -1)

        # load image
        rgb, mask, depth, normal = self.load_images(self.opt.image)
        # print(rgb.shape, mask.shape, depth.shape, normal.shape)
        # exit()

        data = {
            'H': H,
            'W': W,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'dir': dirs.unsqueeze(0),
            'mvp': mvp.unsqueeze(0),
            'polar': self.opt.default_polar,
            'azimuth': self.opt.default_azimuth,
            'radius': self.opt.default_radius,
            'dirkey': [self.id_dir_map[dirs]],
            'camera_type': ["body"],
            'rgb': rgb.unsqueeze(0).to(self.device),
            'mask': mask.to(self.device),
            'depth': depth.to(self.device),
            'normal': normal.unsqueeze(0).to(self.device),
        }

        return data

    def __getitem__(self, idx):
        if self.training:
            # random pose on the fly
            if self.full_body:
                camera_type = "body"
                # poses, dirs, thetas, phis, radius = rand_poses(
                #     1,
                #     self.device,
                #     return_dirs=self.opt.dir_text,
                #     radius_range=self.opt.radius_range,
                #     phi_range=self.opt.phi_range,
                #     theta_range=self.opt.theta_range,
                #     angle_overhead=self.opt.angle_overhead,
                #     angle_front=self.opt.angle_front,
                #     jitter=self.opt.jitter_pose,
                #     uniform_sphere_rate=self.opt.uniform_sphere_rate)
                poses, dirs, thetas, phis, radius = near_head_poses(
                    1,
                    self.device,
                    return_dirs=self.opt.dir_text,
                    radius_range=self.opt.radius_range,
                    phi_range=self.opt.phi_range,
                    theta_range=self.opt.theta_range,
                    angle_overhead=self.opt.angle_overhead,
                    angle_front=self.opt.angle_front,
                    jitter=self.opt.jitter_pose,
                    shift=self.body_center,
                    face_scale=self.body_scale
                )

            else:
                camera_type = "face"
                poses, dirs, thetas, phis, radius = near_head_poses(
                    1,
                    self.device,
                    return_dirs=self.opt.dir_text,
                    phi_range=self.opt.head_phi_range,
                    theta_range=self.opt.head_theta_range,
                    angle_overhead=self.opt.angle_overhead,
                    angle_front=self.opt.angle_front,
                    jitter=self.opt.jitter_pose,
                    shift=self.face_center,
                    face_scale=self.face_scale
                )

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
        else:
            camera_type = "body"
            # circle pose
            radius = 1.3
            fov = 50
            phis = (idx / self.size) * 360
            thetas = 75
            poses, dirs = circle_poses(
                self.device,
                radius=radius,
                theta=thetas,
                phi=phis,
                return_dirs=self.opt.dir_text,
                angle_overhead=self.opt.angle_overhead,
                angle_front=self.opt.angle_front)

            # fixed focal
            # fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2 * focal / self.W, 0, 0, 0],
            [0, -2 * focal / self.H, 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near),
             -(2 * self.far * self.near) / (self.far - self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device)  # yapf: disable

        mvp = projection @ torch.inverse(poses.squeeze(0))

        # sample a low-resolution but full image
        rays_o, rays_d = get_rays(poses, intrinsics, self.H, self.W, -1)

        delta_polar = thetas - self.opt.default_polar
        delta_azimuth = phis - self.opt.default_azimuth
        if delta_azimuth > 180:
            delta_azimuth -= 360  # range in [-180, 180]
        delta_radius = radius - self.opt.default_radius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays_o.squeeze(0),
            'rays_d': rays_d.squeeze(0),
            'dir': dirs,
            'mvp': mvp,  # [4, 4]
            'poses': poses.squeeze(0),
            'intrinsics': torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device),
            'dirkey': self.id_dir_map[dirs],
            # 'azimuth': phis,
            'camera_type': camera_type,

            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }

        return data

    def __len__(self):
        return self.size
