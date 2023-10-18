import torch

def linear_blend_skinning(points, weight, joint_transform, return_vT=False, inverse=False):
    """
    Args:
         points: FloatTensor [batch, N, 3]
         weight: FloatTensor [batch, N, K]
         joint_transform: FloatTensor [batch, K, 4, 4]
         return_vT: return vertex transform matrix if true
         inverse: bool inverse LBS if true
    Return:
        points_deformed: FloatTensor [batch, N, 3]
    """
    if not weight.shape[0] == joint_transform.shape[0]:
        raise AssertionError('batch should be same,', weight.shape, joint_transform.shape)

    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(weight):
        weight = torch.as_tensor(weight).float()
    if not torch.is_tensor(joint_transform):
        joint_transform = torch.as_tensor(joint_transform).float()

    batch = joint_transform.size(0)
    vT = torch.bmm(weight, joint_transform.contiguous().view(batch, -1, 16)).view(batch, -1, 4, 4)
    if inverse:
        vT = torch.inverse(vT.view(-1, 4, 4)).view(batch, -1, 4, 4)

    R, T = vT[:, :, :3, :3], vT[:, :, :3, 3]
    deformed_points = torch.matmul(R, points.unsqueeze(-1)).squeeze(-1) + T

    if return_vT:
        return deformed_points, vT
    return deformed_points



def warp_points(points, skin_weights, joint_transform, inverse=False):
    """
    Warp a canonical point cloud to multiple posed spaces and project to image space
    Args:
        points: [N, 3] Tensor of 3D points
        skin_weights: [N, J]  corresponding skinning weights of points
        joint_transform: [B, J, 4, 4] joint transform matrix of a batch of poses
    Returns:
        posed_points [B, N, 3] warpped points in posed space
    """

    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(joint_transform):
        joint_transform = torch.as_tensor(joint_transform).float()
    if not torch.is_tensor(skin_weights):
        skin_weights = torch.as_tensor(skin_weights).float()

    batch = joint_transform.shape[0]
    if points.dim() == 2:
        points = points.expand(batch, -1, -1)
    # warping
    points_posed, vT = linear_blend_skinning(points,
                                             skin_weights.expand(batch, -1, -1),
                                             joint_transform, return_vT=True, inverse=inverse)

    return points_posed
