from  .args.train_args import get_infer_params 
from .trainers.trainer_factory import load_trainer
import torch 
from .train import make_infer_loader,make_data_loaders
import os 
from .models.model import load_model

def main():
    conf = get_infer_params()
    trainer_func = load_trainer(conf)
    model = load_model(conf)
    dls = make_infer_loader(conf)
    model.load_state_dict(ckpt['model_weights']) 
    old_conf = ckpt['conf']
    old_conf['device'] = conf['device']
    model = model.to(conf['device'][0])
    trainer = trainer_func(model=model,conf=old_conf,loader_dict=dls,tb_writter=None,mode='infer')
    test_csv = trainer.infer_model() 
    orig_pred_dir = conf['output_dir']
    os.makedirs(orig_pred_dir,exist_ok=True)
    test_csv.to_csv(os.path.join(orig_pred_dir,'ts_preds.csv'))



if __name__=='__main__':
    main()
