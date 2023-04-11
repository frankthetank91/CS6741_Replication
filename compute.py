from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from model.preprocessing.joints2jfeats import Rifke
# from temos.tools.geometry import matrix_of_angles
from model.model_utils import remove_padding


# Added
def matrix_of_angles(cos, sin, inv=False, dim=2):
    assert dim in [2, 3]
    sin = -sin if inv else sin
    if dim == 2:
        row1 = torch.stack((cos, -sin), axis=-1)
        row2 = torch.stack((sin, cos), axis=-1)
        return torch.stack((row1, row2), axis=-2)
    elif dim == 3:
        row1 = torch.stack((cos, -sin, 0*cos), axis=-1)
        row2 = torch.stack((sin, cos, 0*cos), axis=-1)
        row3 = torch.stack((0*sin, 0*cos, 1+0*cos), axis=-1)
        return torch.stack((row1, row2, row3),axis=-2)

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)


class ComputeMetrics(Metric):
    def __init__(self, jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if jointstype != "mmm":
            raise NotImplementedError("This jointstype is not implemented.")

        super().__init__()
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype,
                           normalization=False)

        self.force_in_meter = force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # APE
        self.add_state("APE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("APE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints"]

        # AVE
        self.add_state("AVE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("AVE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

    def compute(self):
        count = self.count
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}

        # Compute average of APEs
        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count
        
        # Remove arrays
        APE_metrics.pop("APE_pose")
        APE_metrics.pop("APE_joints")
        
        count_seq = self.count_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        # Remove arrays
        AVE_metrics.pop("AVE_pose")
        AVE_metrics.pop("AVE_joints")
        
        return {**APE_metrics, **AVE_metrics}

    def update(self, jts_text: Tensor, jts_ref: Tensor, lengths: List[int]):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
        jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)

        for i in range(len(lengths)):
            self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
            self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
            self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
            self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

            root_sigma_text = variance(root_text[i], lengths[i], dim=0)
            root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
            self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

            traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
            traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
            self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

            poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
            poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
            self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

            jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
            jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
            self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]], rotations)
        poses = torch.stack((poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat((trajectory[..., :, [0]],
                          root_y[..., None],
                          trajectory[..., :, [1]]), dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]

        if self.force_in_meter:
            # return results in meters
            return (remove_padding(poses / 1000, lengths),
                    remove_padding(poses_local / 1000, lengths),
                    remove_padding(root / 1000, lengths),
                    remove_padding(trajectory / 1000, lengths))
        else:
            return (remove_padding(poses, lengths),
                    remove_padding(poses_local, lengths),
                    remove_padding(root, lengths),
                    remove_padding(trajectory, lengths))
