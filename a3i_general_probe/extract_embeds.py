from .configs.args import get_feat_extract_args # need to implement 
from .helpers.transforms import gen_infer_transforms  
from torch.utils.data import DataLoader
from .models.model_factory import model_loader
from .datasets.mammo_dataset import get_dataset
from .extractors.feat_extractor import ExtractorRegister
import torch
from tqdm import tqdm
import os
import pandas as pd
torch.multiprocessing.set_sharing_strategy('file_system')


def main(): 
    conf = get_feat_extract_args()
    transforms = gen_infer_transforms(conf)

    

if __name__=='__main__':
    main()
