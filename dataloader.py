import torch
import pandas 
import numpy as np
import os
from glob import glob
import json
from pathlib import Path
from typing import Optional, List, Dict
from model.preprocessing.joints2jfeats import Joints2Jfeats, Rifke
from sampling import FrameSampler
from model.preprocessing.xyztransform import XYZTransform
from torch.utils.data import DataLoader



def load_annotation(keyid, datapath):
    metapath = Path(datapath) / (keyid + "_meta.json")
    metadata = json.load(metapath.open())

    if metadata["nb_annotations"] == 0:
       
        return None, False
    annpath = Path(datapath) / (keyid + "_annotations.json")
    anndata = json.load(annpath.open())
    assert len(anndata) == metadata["nb_annotations"]
    return anndata, True


def load_mmm_keyid(keyid, datapath):
    xyzpath = Path(datapath) / (keyid + "_fke.csv")
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

def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")

def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames



def collate_datastruct_and_text(lst_elements: List) -> Dict:
    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate

    batch = {
        # Collate with padding for the datastruct
        "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        # Collate the text
        "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch

def load_text(text_data, keyid, split):
        sequences = text_data[keyid]
        n = len(sequences)
        if split != "test":
            index = np.random.randint(n)
        else:
            # Only the first one in evaluation
            index = 0
        text = sequences[index]
        return text
    
    
    

def collate_datastruct_and_text(lst_elements: List) -> Dict:
    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate

    batch = {
        # Collate with padding for the datastruct
        "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        # Collate the text
        "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch    

def processdata(kit_split,dataset,split):
    keyids = get_split_keyids(kit_split,split)

    enumerator = enumerate(keyids)
    transform = XYZTransform(joints2jfeats=Joints2Jfeats)
    feature_data ={}
    text_data ={}
    durations ={}
    ELEM=[]
    for i, keyid in enumerator:
        annotation, success = load_annotation(keyid,dataset)
        joints = load_mmm_keyid(keyid,dataset)
        joints, duration = downsample_mmm(joints,downsample = True,framerate=12.5)
        
        features = Rifke(path = './dependencies/transforms/kit-mmm-xyz',normalize=True).forward(joints=joints)
        feature_data[keyid] = features
        text_data[keyid] = annotation
        durations[keyid] = duration
        frame_ix = FrameSampler(duration)
        
        datastruct = transform.Datastruct(features=features)
        element = {"datastruct":datastruct , "text": annotation, "length": len(datastruct), "keyid" : keyid}
        ELEM.append(element)
    return ELEM    
         

def make_dataloader(kit_split, dataset, split,shuffle=True):
    keyid = get_split_keyids(kit_split,split)
    dataloader_option = {"batch_size":8, "num_workers" : 8, "collate_fn":collate_datastruct_and_text}
    elem = processdata(kit_split,dataset,split)
    return DataLoader(elem,shuffle=shuffle,**dataloader_option)

if __name__ == "__main__":
    kit_split = './dataset/kit-splits'
    dataset = './dataset/dataset_raw'
    
    train_keyid = get_split_keyids(kit_split,'train')
    val_keyid = get_split_keyids(kit_split,'val')
    test_keyid = get_split_keyids(kit_split,'test')
    dataloader_option = {"batch_size":8, "num_workers" : 8, "collate_fn":collate_datastruct_and_text}
    
    elem = processdata(kit_split,dataset,'train')
    train_dataloader = DataLoader(elem,shuffle=True,**dataloader_option)