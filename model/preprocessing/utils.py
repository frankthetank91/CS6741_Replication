from typing import Optional, List, Dict
from torch import Tensor
import numpy as np
from pathlib import Path

def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")


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

def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames

def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas



def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {"length": [x["length"] for x in lst_elements],
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch and x != "datastruct"]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch
