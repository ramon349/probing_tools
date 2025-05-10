from ..modules.efficientnet_custom import EfficientNet
from torch import nn
import torch
from collections import OrderedDict
from vector_quantize_pytorch import VectorQuantize
from ..modules import load_projection_head 
from torch.nn import functional as F 
from torch.autograd import Function  

class ModelRegister:
    __data = {}

    @staticmethod
    def __models():
        if not hasattr(ModelRegister, "_data"):
            ModelRegister._data = {}
        return ModelRegister._data

    @classmethod
    def register(cls, cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name] = cls_obj
            return cls_obj

        return decorator

    @classmethod
    def get_model(cls, key):
        return cls.__data[key]

    @classmethod
    def num_models(cls):
        return len(cls.__data)

    @classmethod
    def get_models(cls):
        return cls.__data.keys()


def model_loader(conf):
    model_params = conf["model_parameters"]
    model = ModelRegister.get_model(conf["model"])(model_params)
    if "model_weight" in conf and conf["model_weight"]:
        model_w = torch.load(conf["model_weight"], map_location="cpu")
        model_w = model_w["model_weights"]
        model_w = remove_module(model_w)
        model.load_state_dict(model_w)
        print(f"Loaded state dict")
    model = model.to(conf["device"][0])
    return model


def remove_module(w_d):
    new_d = OrderedDict()
    for k, v in w_d.items():
        new_name = k.replace("module.", "")
        new_d[new_name] = v
    return new_d


@ModelRegister.register(cls_name="mammo_clip_vision")
class VisionEnc(nn.Module):
    def __init__(self, config) -> None:
        super(VisionEnc, self).__init__()
        weight_path = config['w_path']
        self.encoder = self.load_encoder(weight_path) 

    def load_encoder(self, w_path):
        model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        model.out_dim = 2048
        weights = self._weight_proc(w_path)
        model.load_state_dict(weights)
        return model

    def _weight_proc(self, weight_path):
        config = torch.load(weight_path, map_location="cpu",weights_only=False)
        image_encoder_weights = {}
        for k in config["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = config["model"][k]
        return image_encoder_weights

    def forward(self, x):
        return self.encoder(x)
    def _get_embedding(self,x): 
        return self.encoder(x)

@ModelRegister.register(cls_name="LinearProbe")
class VisionProbe(nn.Module): 
    def __init__(self,config ) -> None:
        super(VisionProbe, self).__init__()
        num_task = config['n_classes']
        self.cls = nn.Linear(2048,num_task)
    def forward(self, x):
        return  self.cls(x) 

