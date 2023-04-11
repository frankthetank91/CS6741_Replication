import torch
from dataclasses import dataclass
"""
This part converts original motion capture data into pytorch dataset style class

-> The original file (.csv) contains xyz coordinate of each joint




"""







import json
import os
from glob import glob
from typing import Dict, Optional
import logging

import numpy as np
import pandas
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path



from rich.progress import track
        
from .base import BASEDataModule
from .utils import get_split_keyids
from .transforms import Transform
from .xyztransform import XYZTransform
from .utils import subsample
from .joints2jfeats import Joints2Jfeats

from sampling import FrameSampler

class KITDataModule(BASEDataModule):
    def __init__(self, data_dir: str = "./dataset/dataset_raw",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)

        self.Dataset = KIT

        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = 64
        self.transforms = XYZTransform


class KIT(Dataset):
    dataname = "KIT Motion-Language"

    def __init__(self, datapath: str = "./dataset/dataset_raw",
                 splitpath: str = './dataset/kit-splits',
                 transforms: Transform = XYZTransform ,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 sampler=FrameSampler,
                 framerate: float = 12.5,
                 pick_one_text: bool = True,
                 downsample=True,
                 tiny: bool = False, **kwargs):

        self.split = split
        self.downsample = downsample


        self.transforms = transforms
        self.sampler = sampler
        self.pick_one_text = pick_one_text

        super().__init__()
        keyids = get_split_keyids(path=splitpath, split=split)

        features_data = {}
        texts_data = {}
        durations = {}
        enumerator = enumerate(keyids)
        maxdata = np.inf
        datapath = Path(datapath)
        num_bad = 0
  
        for i, keyid in enumerator:
            if len(features_data) >= maxdata:
                break

            anndata, success = load_annotation(keyid, datapath)

            joints = load_mmm_keyid(keyid, datapath)
            joints, duration = downsample_mmm(joints, downsample=self.downsample, framerate=framerate)

            if split != "test" and not tiny:
                # Accept or not the sample, based on the duration
                if not self.sampler.accept(duration):
                    num_bad += 1
                    continue

            # Load rotation features (rfeats) data from AMASS


            features = Joints2Jfeats(joints)

            features_data[keyid] = features
            texts_data[keyid] = anndata
            durations[keyid] = duration
            
        if split != "test" and not tiny:
            total = len(features_data)
            percentage = 100 * num_bad / (total+num_bad)
       
        self.features_data = features_data
        self.texts_data = texts_data

        self.keyids = list(features_data.keys())
        self._split_index = list(self.keyids)
        self._num_frames_in_sequence = durations
        self.nfeats = len(self[0]["datastruct"].features[0])

    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.features_data[keyid]
        datastruct = self.transforms.Datastruct(self,features=features)
        return datastruct

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        if not self.pick_one_text:
            return sequences
        n = len(sequences)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            # Only the first one in evaluation
            index = 0
        text = sequences[index]
        return text

    def load_keyid(self, keyid):
        num_frames = self._num_frames_in_sequence[keyid]
        frame_ix = self.sampler(num_frames)

        datastruct = self._load_datastruct(keyid, frame_ix)
        text = self._load_text(keyid)
        element = {"datastruct": datastruct, "text": text,
                   "length": len(datastruct), "keyid": keyid}
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"


def load_annotation(keyid, datapath):
    metapath = datapath / (keyid + "_meta.json")
    metadata = json.load(metapath.open())

    if metadata["nb_annotations"] == 0:
       
        return None, False
    annpath = datapath / (keyid + "_annotations.json")
    anndata = json.load(annpath.open())
    assert len(anndata) == metadata["nb_annotations"]
    return anndata, True


def load_mmm_keyid(keyid, datapath):
    xyzpath = datapath / (keyid + "_fke.csv")
    xyzdata = pandas.read_csv(xyzpath, index_col=0)
    joints = np.array(xyzdata).reshape(-1, 21, 3)
    return joints

def downsample_mmm(joints, *, downsample, framerate):
    nframes_total = len(joints)
    last_framerate = 100

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)
    joints = torch.from_numpy(joints[frames]).float()
    return joints, duration

