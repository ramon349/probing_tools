import torchvision.transforms as torch_trx 
from skimage.measure import label, regionprops_table
import pandas as pd
import torch
import numpy as np

class TransformFactory: 
    """ The purpose of this class is to be a global dictionary that stores  transforms 
    The register method is a decoractor function that will add a function to a list of avialable transforms
    """
    _transforms = {}
    @classmethod
    def register(cls,cls_name=None):
        def decorator(cls_obj):
            cls._transforms[cls_name]=cls_obj
            return cls_obj
        return decorator
    @classmethod
    def get_transform(cls,key,config): 
        transform_func = cls._transforms[key]
        return  transform_func(config) # This initializes the transform
    @classmethod
    def num_datasets(cls):
        return len(cls._transforms)
    @classmethod
    def get_datasets(cls):
        return cls._transforms.keys()

def get_transform(transform_name,config=None):
    return TransformFactory.get_transform(transform_name,config=config)
@TransformFactory.register("crop")
class CropMammo(object):
    def __init__(self,config) -> None:
        super(CropMammo, self).__init__()

    def __call__(self, tens):
        arr = tens.numpy()
        zero_mask = arr > arr.min()
        lbl = label(zero_mask)
        props = pd.DataFrame(
            regionprops_table(lbl, properties=["label", "bbox", "area"])
        )
        props = props.sort_values(by="area", ascending=False)
        row = props.iloc[0]
        rmin, rmax, cmin, cmax = row[[f"bbox-{e}" for e in [1, 4, 2, 5]]].astype(
            int
        )  # BEWARE OF THE BATCH DIMENSION
        new_tens = tens[:, rmin:rmax, cmin:cmax]
        return new_tens
@TransformFactory.register("WindowHU")
class WindowHU(object):
    def __init__(self,config,w_min=-150,w_max=200,bit_depth=8) -> None:
        self.w_min = w_min 
        self.w_max = w_max
        self.bit_depth = bit_depth
        assert bit_depth in [8,16]
    def __call__(self,tens): 
        tens[tens<self.w_min]=self.w_min 
        tens[tens>self.w_max]=self.w_max
        tens = tens + abs(tens.min())
        tens = tens/tens.max() 
        bit_val = (2**self.bit_depth)-1
        tens = (tens*(bit_val)).astype(np.uint16)
        return tens 
@TransformFactory.register("make8bit")
class MakeUint(object): 
    def  __init__(self,config,bitdepth=8):
        self.bitdepth = bitdepth
    def __call__(self, tens): 
        tens = tens + abs(tens.min())
        tens = tens/tens.max() 
        bit_val = (2**self.bitdepth)-1
        tens = (tens*(bit_val))
        return tens 
@TransformFactory.register("makeZeroOne")
class MakeZeroOne(object):
    def __init__(self,config=None):
        super(MakeZeroOne,self).__init__()
    def __call__(self, x):
        normed = x/x.max() 
        return normed

@TransformFactory.register("grey2rgb")
class Grey2RGB(object):
    def __init__(self,config=None) -> None:
        super(Grey2RGB, self).__init__()

    def __call__(self, x):
        return x.repeat(3, 1, 1).to(torch.float)
@TransformFactory.register("stack")
class StackTens(object): 
    def __init__(self,config,name='stack3D'):
        self.name = name
    def __call__(self, arr):
        match type(arr): 
            case np.ndarray: 
                return np.dstack((arr,arr,arr)).astype(float)
            case torch.Tensor: 
                return torch.vstack((arr,arr,arr)).to(torch.float)
@TransformFactory.register("norm")
def _gen_normalize(config): 
    mu = config['norm_mu']=config['norm_mu']
    std = config['norm_std']=config['norm_std']
    return torch_trx.Normalize(mu,std)

@TransformFactory.register("tensor")
def _to_tensor(config): 
    return torch_trx.ToTensor()

@TransformFactory.register("resize")
def _gen_resize(config): 
    shape0 = config['img_shape'][0]
    shape1 = config['img_shape'][1]
    return torch_trx.Resize((shape0,shape1))

@TransformFactory.register("centerCrop") 
def _gen_center_crop(config): 
    return torch_trx.CenterCrop((224,224))

@TransformFactory.register("toTensor")
def _gen_to_tensor(config):
    return torch_trx.ToTensor() 

def gen_transforms(confi):
    transform_conf = confi['transform_conf']
    train_transform = torch_trx.Compose(
        [TransformFactory.get_transform(e, transform_conf) for e in confi["train_transforms"]]
    )
    val_transform = torch_trx.Compose([TransformFactory.get_transform(e, transform_conf) for e in confi["test_transforms"]])
    return train_transform, val_transform


def gen_infer_transforms(confi):
    my_transforms = list() 
    for e in confi['test_transforms']:
        l_transform = get_transform(e,confi['transform_conf'])
        my_transforms.append(l_transform)
    val_transform = torch_trx.Compose(my_transforms) #Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return val_transform

