import pandas as pd 
from  .configs.args import get_train_args 
from  .trainers.trainer_factory import load_trainer
from .datasets.mammo_dataset import get_dataset
from .models.foundation_model import model_loader 
from torch.utils.data import DataLoader
from glob import glob 
import os 
import torch 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler 
torch.multiprocessing.set_sharing_strategy('file_system')

def make_data_loaders(conf): 
    dl_dict = {}  
    batch_size = conf['batch_size']
    dsobj = get_dataset(conf) 
    debug = True if 'debug' in conf and conf['debug'] else False
    for n,e in zip(["train","val","ts"],['internal-train','internal-val','internal-test']): 
        ds = dsobj(conf['csv_path'],split=e,conf=conf) 
        if debug: 
            ds.df = ds.df.sample(frac=0.25,random_state=1996)
        if n=='train': 
            sampler = ds._make_weighted_sampler() #UniquePatientSampling(ds.df,num_samples=ds.df.shape[0],bs=40)
            shuffle = False 
        else: 
            sampler = None 
            shuffle =True
        dl_dict[n] = DataLoader(ds,batch_size= batch_size,persistent_workers=True,num_workers=16,shuffle=shuffle,sampler=sampler,drop_last=n=='train')
    return dl_dict
def make_infer_loader(conf): 
    dl_dict = {}  
    batch_size = conf['batch_size']
    dsobj = load_dataset(conf) 
    ds = dsobj(conf['csv_path'],split='infer',conf=conf) 
    dl_dict['infer'] = DataLoader(ds,batch_size= batch_size,persistent_workers=True,num_workers=16,shuffle=True)
    return dl_dict

def make_writer(conf): 
    log_path = figure_version(
        conf["log_dir"]
    )  
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    writer =  SummaryWriter(log_dir=log_path)
    return writer,log_path
def figure_version(path: str, load_past=False):
    #  when saving model  checkpoints and logs. Need to make sure i don't overwrite previous experiemtns
    log_dir = path 
    log_files = glob(os.path.join(log_dir,'events*'))
    for e in log_files:
        os.remove(e)
    return log_dir 



def main():
    conf = get_train_args()
    trainer_func = load_trainer(conf) 
    model = model_loader(conf)
    dls = make_data_loaders(conf)
    log_writer,log_dir = make_writer(conf)
    conf['log_dir'] = log_dir
    trainer = trainer_func(model=model,conf=conf,loader_dict=dls,tb_writter=log_writer)
    trainer.fit()
    conf['model_weights'] = os.path.join(conf['log_dir'],'model_w.ckpt')
    ckpt = torch.load(conf['model_weights'],map_location='cpu') 
    model.load_state_dict(ckpt['model_weights']) 
    for split in ['ts','val']:
        test_csv = trainer.test_model(split=split) 
        test_path = os.path.join(log_dir,f'{split}_preds.csv')
        test_csv.to_csv(test_path,index=False)

if __name__=='__main__': 
    main() 
