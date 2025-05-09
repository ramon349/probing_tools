import pandas as pd
import pydicom as pyd
from pydicom.pixel_data_handlers import apply_voi_lut
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch 

class DataRegister:
    """ The purpose of this class is to be a global dictionary that stores all the dataset classes 
    The register method is a decoractor function that will add the class function to a list of avialable datasets 

    """
    __data = {}

    @staticmethod
    def __datasets():
        if not hasattr(DataRegister, "_data"):
            DataRegister._data = {}
        return DataRegister._data

    @classmethod
    def register(cls, cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name] = cls_obj
            return cls_obj

        return decorator

    @classmethod
    def num_datasets(cls):
        return len(cls.__data)

    @classmethod
    def get_dataset(cls, key):
        return cls.__data[key]

    @classmethod
    def get_datasets(cls):
        return cls.__data.keys()


def get_dataset(conf):
    """ Looks at our dataset dictionary and returns the match if there is one 
    """ 
    dset_name = conf['dataset'] 
    try: 
       dset= DataRegister.get_dataset(conf["dataset"])
    except KeyError: 
        avail_dsets = DataRegister.get_datasets()
        raise KeyError(f"You requested {dset_name}. However we only have {avail_dsets}") 
    return dset 


def load_dcm(dcm_path, apply_voi=True): 
    """ Takes a dcm and applies an optinal voi transformation. 
    Returns numpy array contained in the image 
    """
    dcm_path = dcm_path
    dcm = pyd.dcmread(dcm_path)
    arr = dcm.pixel_array
    if apply_voi:
        windowed_arr = apply_voi_lut(arr, dcm, prefer_lut=True)
        return windowed_arr
    else:
        return arr

def load_png(png_path):
    """ Loads a png/jpeg image and returns numpy array
    """
    with Image.open(png_path, "r") as f:
        arr = np.array(f)
    return arr
@DataRegister.register("MammoFeatExtract")
class MammoFeatExtract(Dataset):
    """ Dataset class used for extraction.
    Uses a csv file to find the path to images. 
    Can load either png/jpeg images or dicom files 
    """
    def __init__(self, conf=None,transforms=None) -> None:
        super().__init__()
        csv_path = conf["csv_path"]
        self.load_func = self.get_img_func(conf)
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        col_info = conf["col_info"]
        self.col_info = col_info
        self.img_col = self.col_info["img_path"]
    def get_img_func(self, conf):
        func_name = conf["image_func"]
        if func_name == "dcm":
            return load_dcm
        if func_name == "png":
            return load_png
    def __getitem__(self, index):
        df_row = self.df.iloc[index]
        img_path = df_row[self.img_col].strip(" ")
        print(img_path)
        img = self.load_func(img_path)
        if self.transforms:
            img = self.transforms(img).to(torch.float)
        return img, img_path
    def __len__(self):
        return self.df.shape[0]
