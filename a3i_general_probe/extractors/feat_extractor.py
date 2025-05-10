import torch 
from tqdm import tqdm
from  hashlib import sha224
import os 
import pandas as pd 
class ExtractorRegister: 
    __data = {} 
    @classmethod 
    def register(cls,cls_name=None): 
        def decorator(cls_obj):
            cls.__data[cls_name]=cls_obj
            return cls_obj 
        return decorator
    @classmethod 
    def get_extractor(cls,key):
        return cls.__data[key]
def load_extractor(conf): 
    return ExtractorRegister.get_extractor(conf) 
@ExtractorRegister.register("MammoClip")
class MammoClipExtractor(): 
    def __init__(self,config,model,data_loader): 
        self.config = config
        self.device = self.config['device'][0]
        self.output_dir = self.config['output_dir']
        self.batch_size = self.config['batch_size']
        self.img_col = self.config['col_info']['img_path']
        self.dl = data_loader
        self.model = model 
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model) 
    def _get_hash(self,fpath):
        my_hash = sha224(fpath.encode('utf-8')).hexdigest()
        return my_hash 
    def make_embed_name(self,fpath): 
        hashname = self._get_hash(fpath)
        output_path = os.path.join(self.output_dir,f"{hashname}.pt")
        return output_path

    def _get_embedding(self,img_batch): 
        img_batch = img_batch.to(self.device)
        img_feats = self.model.encode_image(img_batch)
        return img_feats.cpu() 
    def run_extract(self): 
        gt_paths,pred_paths = list(),list() 
        for batch in tqdm(self.dl,total=len(self.dl)):
            img,img_path = batch
            img = img.to(self.device) 
            with torch.inference_mode(True): 
                img_feats = self.model._get_embedding(img).cpu()
            for i,path in enumerate(img_path):
                save_path = self.make_embed_name(path)
                gt_paths.append(path)
                pred_paths.append(save_path)
                sample = img_feats[i,:]
                torch.save(sample,save_path)
        return pd.DataFrame({self.img_col:gt_paths,"feat_path":pred_paths} )
