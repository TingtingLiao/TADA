import os
import random
import json
import pickle as pkl
import cv2
import numpy as np
import imageio
import torch
from packaging import version as pver

from yacs.config import CfgNode as CN


def load_config(path, default_path=None):
    cfg = CN(new_allowed=True)
    if default_path is not None:
        cfg.merge_from_file(default_path)
    cfg.merge_from_file(path)

    return cfg


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def plot_grid_images(images, row, col, save_path=None):
    """
    Args:
        images: np.array [B, H, W, 3]
        row:
        col:
        save_path:

    Returns:

    """
    assert row * col == images.shape[0]
    images = np.vstack([np.hstack(images[r * col:(r + 1) * col]) for r in range(row)])
    if save_path:
        cv2.imwrite(save_path, images * 255)
    return images


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W], [H, W, 3] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return rays_o, rays_d


def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[
        1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def save_image(fn, x: np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8),
                            compress_level=3)  # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)


# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1 / (y * aspect), 0, 0, 0],
                         [0, 1 / -y, 0, 0],
                         [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                         [0, 0, -1, 0]], dtype=torch.float32, device=device)


def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1, 0, 0, 0],
                         [0, c, s, 0],
                         [0, -s, c, 0],
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[c, 0, s, 0],
                         [0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)


@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


class SMPLXSeg:
    smplx_dir = "./data/smplx"
    smplx_segs = json.load(open(f"{smplx_dir}/smplx_vert_segementation.json"))
    flame_segs = pkl.load(open(f"{smplx_dir}/FLAME_masks.pkl", "rb"), encoding='latin1')
    smplx_face = np.load(f"{smplx_dir}/smplx_faces.npy")

    # 'eye_region',  , 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
    # 'rightHand', 'rightUpLeg', 'leftArm', 'head',
    # 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot',
    # 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm',
    # 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck',
    # 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips']
    # print(smplx_segs.keys())
    # exit()

    smplx_flame_vid = np.load(f"{smplx_dir}/FLAME_SMPLX_vertex_ids.npy", allow_pickle=True)

    eyeball_ids = smplx_segs["leftEye"] + smplx_segs["rightEye"]
    hands_ids = smplx_segs["leftHand"] + smplx_segs["rightHand"] + \
                smplx_segs["leftHandIndex1"] + smplx_segs["rightHandIndex1"]
    neck_ids = smplx_segs["neck"]
    head_ids = smplx_segs["head"]

    front_face_ids = list(smplx_flame_vid[flame_segs["face"]])
    ears_ids = list(smplx_flame_vid[flame_segs["left_ear"]]) + list(smplx_flame_vid[flame_segs["right_ear"]])
    forehead_ids = list(smplx_flame_vid[flame_segs["forehead"]])
    lips_ids = list(smplx_flame_vid[flame_segs["lips"]])
    nose_ids = list(smplx_flame_vid[flame_segs["nose"]])
    eyes_ids = list(smplx_flame_vid[flame_segs["right_eye_region"]]) + list(
        smplx_flame_vid[flame_segs["left_eye_region"]])
    check_ids = list(
        set(front_face_ids) - set(forehead_ids) - set(lips_ids) - set(nose_ids) - set(eyes_ids)
    )

    # re-mesh mask
    remesh_ids = list(set(front_face_ids) - set(forehead_ids)) + ears_ids + eyeball_ids + hands_ids
    remesh_mask = ~np.isin(np.arange(10475), remesh_ids)
    remesh_mask = remesh_mask[smplx_face].all(axis=1)


def create_checkerboard(h, w, c, grid_size):
    num_grid_row = h // grid_size
    num_grid_col = w // grid_size
    grid_ones = np.ones((grid_size, grid_size, c))
    grid_zeros = np.zeros((grid_size, grid_size, c))

    checkerboard = np.vstack([
        np.hstack([grid_ones if (c + r) % 2 == 1 else grid_zeros for c in range(num_grid_col)])
        for r in range(num_grid_row)
    ])

    # pad
    cx, cy, _ = checkerboard.shape
    out = np.ones((h, w, c))
    dx = (h - cx) // 2
    dy = (w - cy) // 2
    out[dx:dx + cx, dy:dy + cy] = checkerboard
    return out


if __name__ == '__main__':
    out = create_checkerboard(512, 512, 3, 64)
    import cv2

    cv2.imwrite("ck.png", out * 255)
