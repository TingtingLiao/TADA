import os
import cv2
import torch
import numpy as np
import pymeshlab
import trimesh
from .utils import dot, safe_normalize


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def compute_normal(vertices, faces):
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices).float()
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(faces).long()

    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()

    v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    vn = torch.zeros_like(vertices)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    vn = torch.where(dot(vn, vn) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
    vn = safe_normalize(vn)

    face_normals = safe_normalize(face_normals)
    return vn, faces


class Mesh():
    def __init__(self, v=None, f=None, vn=None, fn=None, vt=None, ft=None, albedo=None, device=None, base=None,
                 init_empty_tex=False, albedo_res=1024):
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        self.v_tng = None
        self.f_tng = None
        # only support a single albedo
        if init_empty_tex:
            self.albedo = torch.zeros((albedo_res, albedo_res, 3), dtype=torch.float32, device=device)

        else:
            self.albedo = albedo
        self.device = device

        if isinstance(base, Mesh):
            for name in ['v', 'vn', 'vt', 'f', 'fn', 'ft', 'albedo', 'v_tng', 'f_tng']:
                if getattr(self, name) is None:
                    setattr(self, name, getattr(base, name))

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None, init_empty_tex=False, albedo_res=1024,
                 uv_path=None, normalize=False):

        assert os.path.splitext(path)[-1] == '.obj'

        mesh = cls()

        # device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mesh.device = device

        # try to find texture from mtl file
        if albedo_path is None:
            mtl_path = path.replace('.obj', '.mtl')
            if os.path.exists(mtl_path):
                with open(mtl_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0: continue
                    prefix = split_line[0]
                    # NOTE: simply use the first map_Kd as albedo!
                    if 'map_Kd' in prefix:
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f'[load_obj] use albedo from: {albedo_path}')
                        break

        if init_empty_tex or albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f'[load_obj] init empty albedo!')
            albedo = np.ones((albedo_res, albedo_res, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color
        else:
            albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = cv2.resize(albedo, (albedo_res, albedo_res))
            albedo = albedo.astype(np.float32) / 255

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        # load obj
        with open(path, 'r') as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != '' else -1 for x in fv.split('/')]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0: continue
            # v/vn/vt
            prefix = split_line[0].lower()
            if prefix == 'v':
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == 'vn':
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == 'vt':
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == 'f':
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = torch.tensor(texcoords, dtype=torch.float32, device=device) if len(texcoords) > 0 else None
        mesh.vn = torch.tensor(normals, dtype=torch.float32, device=device) if len(normals) > 0 else None

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = torch.tensor(tfaces, dtype=torch.int32, device=device) if texcoords is not None else None
        mesh.fn = torch.tensor(nfaces, dtype=torch.int32, device=device) if normals is not None else None

        # auto-normalize
        # Skip this
        if normalize:
            mesh.auto_size()

        print(f'[load_obj] v: {mesh.v.shape}, f: {mesh.f.shape}')

        # auto-fix normal
        if mesh.vn is None:
            mesh.auto_normal()

        print(f'[load_obj] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}')

        # auto-fix texture
        if mesh.vt is None:
            mesh.auto_uv(cache_path=uv_path)

        print(f'[load_obj] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}')

        return mesh

    @classmethod
    def load_albedo(cls, albedo_path):
        albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        albedo = albedo.astype(np.float32) / 255
        return albedo

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):  # to [-0.5, 0.5]
        vmin, vmax = self.aabb()
        scale = 1 / torch.max(vmax - vmin).item()
        self.v = self.v - (vmax + vmin) / 2  # Center mesh on origin
        self.v = self.v * scale

    def auto_normal(self):
        self.vn, self.fn = compute_normal(self.v, self.f)
        self.fn = self.f

    @torch.no_grad()
    def auto_uv(self, cache_path="", v=None, f=None):
        # try to load cache
        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np = data['vt'], data['ft']
        else:
            import xatlas
            if v is not None and f is not None:
                v_np = v.cpu().numpy()
                f_np = f.int().cpu().numpy()
            else:
                v_np = self.v.cpu().numpy()
                f_np = self.f.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        # save to cache
        # np.savez(cache_path, vt=vt_np, ft=ft_np)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)

        self.vt = vt
        self.ft = ft
        return vt, ft

    def compute_tangents(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v[self.f[:, i]]
            tex[i] = self.vt[self.ft[:, i]]
            vn_idx[i] = self.fn[:, i]

        tangents = torch.zeros_like(self.vn)
        tansum = torch.zeros_like(self.vn)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
        denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = safe_normalize(tangents)
        tangents = safe_normalize(tangents - dot(tangents, self.vn) * self.vn)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))
        self.v_tng = tangents
        self.f_tng = self.fn

    # write to obj file
    def write(self, path):

        mtl_path = path.replace('.obj', '.mtl')
        albedo_path = path.replace('.obj', '_albedo.png')

        v_np = self.v.cpu().numpy()
        vt_np = self.vt.cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.cpu().numpy() if self.vn is not None else None
        f_np = self.f.cpu().numpy()
        ft_np = self.ft.cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f'mtllib {os.path.basename(mtl_path)} \n')

            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            for v in vt_np:
                fp.write(f'vt {v[0]} {1 - v[1]} \n')

            for v in vn_np:
                fp.write(f'vn {v[0]} {v[1]} {v[2]} \n')

            fp.write(f'usemtl defaultMat \n')
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n')

        with open(mtl_path, "w") as fp:
            fp.write(f'newmtl defaultMat \n')
            fp.write(f'Ka 1 1 1 \n')
            fp.write(f'Kd 1 1 1 \n')
            fp.write(f'Ks 0 0 0 \n')
            fp.write(f'Tr 1 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0 \n')
            fp.write(f'map_Kd {os.path.basename(albedo_path)} \n')

        albedo = self.albedo.detach().cpu().numpy()
        albedo = (albedo * 255).astype(np.uint8)
        cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))

    def set_albedo(self, albedo):
        self.albedo = torch.sigmoid(albedo)

    def set_uv(self, vt, ft):
        self.vt = vt
        self.ft = ft

    def auto_uv_face_att(self):
        import kaolin as kal
        self.uv_face_att = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long())


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def keep_largest(mesh):
    mesh_lst = mesh.split(only_watertight=False)
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh


def poisson(mesh, obj_path):
    mesh.export(obj_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.set_verbosity(False)
    ms.generate_surface_reconstruction_screened_poisson(depth=10, preclean=True)
    ms.set_current_mesh(1)
    ms.save_current_mesh(obj_path)

    new_meshes = trimesh.load(obj_path)
    new_mesh_lst = new_meshes.split(only_watertight=False)
    comp_num = [new_mesh.vertices.shape[0] for new_mesh in new_mesh_lst]

    return new_mesh_lst[comp_num.index(max(comp_num))]


def mesh_clean(mesh, save_path=None):
    """ clean mesh """
    cc = mesh.split(only_watertight=False)
    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1, 0] - bbox[0, 0]
    for c in cc:
        bbox = c.bounds
        if height < bbox[1, 0] - bbox[0, 0]:
            height = bbox[1, 0] - bbox[0, 0]
            out_mesh = c
    if save_path is not None:
        out_mesh.export(save_path)
    return out_mesh


def normalize_vert(vertices, return_cs=False):
    if isinstance(vertices, np.ndarray):
        vmax, vmin = vertices.max(0), vertices.min(0)
        center = (vmax + vmin) * 0.5
        scale = 1. / np.max(vmax - vmin)
    else:  # torch.tensor
        vmax, vmin = vertices.max(0)[0], vertices.min(0)[0]
        center = (vmax + vmin) * 0.5
        scale = 1. / torch.max(vmax - vmin)
    if return_cs:
        return (vertices - center) * scale, center, scale
    return (vertices - center) * scale
