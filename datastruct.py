rom dataclasses import dataclass, fields

from model.preprocessing.utils import collate_tensor_with_padding
from typing import Optional
from torch import Tensor

from model.preprocessing.transforms import Transform , Datastruct
from dataclasses import dataclass, fields
from model.preprocessing.utils import collate_tensor_with_padding

from model.preprocessing.joints2jfeats import Joints2Jfeats

class Transform:
    def collate(self, lst_datastruct):
        
        example = lst_datastruct[0]

        def collate_or_none(key):
            if example[key] is None:
                return None
            key_lst = [x[key] for x in lst_datastruct]
            return collate_tensor_with_padding(key_lst)

        kwargs = {key: collate_or_none(key)
                  for key in example.datakeys}

        return self.Datastruct(**kwargs)


# Inspired from SMPLX library
# need to define "datakeys" and transforms
@dataclass
class Datastruct:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def to(self, *args, **kwargs):
        for key in self.datakeys:
            if self[key] is not None:
                self[key] = self[key].to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self[self.datakeys[0]].device

    def detach(self):
        def detach_or_none(tensor):
            if tensor is not None:
                return tensor.detach()
            return None

        kwargs = {key: detach_or_none(self[key])
                  for key in self.datakeys}
        return self.transforms.Datastruct(**kwargs)
    
    
class XYZTransform(Transform):
    def __init__(self, joints2jfeats: Joints2Jfeats, **kwargs):
        self.joints2jfeats = joints2jfeats

    def Datastruct(self, **kwargs):
        return XYZDatastruct(_joints2jfeats=self.joints2jfeats,
                             transforms=self,
                             **kwargs)

    def __repr__(self):
        return "XYZTransform()"

@dataclass
class XYZDatastruct(Datastruct):
    transforms: XYZTransform
    _joints2jfeats: Joints2Jfeats

    features: Optional[Tensor] = None
    joints_: Optional[Tensor] = None
    jfeats_: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features", "joints_", "jfeats_"]
        # starting point
        if self.features is not None and self.jfeats_ is None:
            self.jfeats_ = self.features

    @property
    def joints(self):
        # Cached value
        if self.joints_ is not None:
            return self.joints_

        # self.jfeats_ should be defined
        assert self.jfeats_ is not None

        self._joints2jfeats.to(self.jfeats.device)
        self.joints_ = self._joints2jfeats.inverse(self.jfeats)
        return self.joints_

    @property
    def jfeats(self):
        # Cached value
        if self.jfeats_ is not None:
            return self.jfeats_

        # self.joints_ should be defined
        assert self.joints_ is not None

        self._joints2jfeats.to(self.joints.device)
        self.jfeats_ = self._joints2jfeats(self.joints)
        return self.jfeats_

    def __len__(self):
        return len(self.jfeats)
