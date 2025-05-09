from .configs.args import get_feat_extract_args # need to implement 
from .datasets.mammo_dataset import get_dataset
from .helpers.transforms import gen_infer_transforms
from torch.utils.data import DataLoader
#from .models.model_factory import model_loader
#from .extractors.feat_extractor import ExtractorRegister
import torch
from tqdm import tqdm
import os
import pandas as pd

def make_data_loader(conf):
    batch_size = conf["batch_size"]
    n_workers = conf["num_workers"]
    val_trx = gen_infer_transforms(conf)
    dataset = get_dataset(conf)
    ts_ds = dataset(transforms=val_trx, conf=conf)
    if conf['debug']==True: 
        ts_ds.df = ts_ds.df.sample(n=100,random_state=1996)
    ts_dl = DataLoader(
            ts_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            persistent_workers=True,
            shuffle=False,
        )
    return ts_dl 

def main(): 
    conf = get_feat_extract_args()
    dl = make_data_loader(conf) 


    

if __name__=='__main__':
    main()
