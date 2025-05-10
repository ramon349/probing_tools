from torch.nn import  CrossEntropyLoss
from torch import optim
from .trainer_factory import TrainerRegister 
from tqdm import tqdm 
import torch 
import os 
from torch.nn.functional import softmax
from collections import defaultdict
import pandas as pd 
from sklearn.metrics import recall_score,precision_score,roc_auc_score
from collections import deque
from torch.nn import BCELoss
import optuna 
import numpy as np 

@TrainerRegister.register(cls_name='LateFusion')
class LateFusionTrainer:
    def __init__(self, model=None, conf=None, loader_dict=None, tb_writter=None,mode='train'):
        self.conf = conf
        self.model = model 
        self.tb: SummaryWriter =   tb_writter
        self.total_epochs = conf['epochs']
        self.loader_d = loader_dict
        self.device = conf['device'][0]
        self._log_model_graph()
        self._build_criterions()
        self.init_optims()
        self.gb_step = 0 
        self.c_epoch =0
        self.disable_bar =False  
        self.mode= mode
        self._log_model_graph()
    def _log_model_graph(self):
        if self.tb:
            sample = next(iter(self.loader_d['train']))[0]
            self.tb.add_graph(self.model,sample.to(self.device)) 
    def _log_scalar(self,name,val,step): 
        if self.tb: 
            self.tb.add_scalar(name,val,global_step=step)
    def init_optims(self): 
        self.opti = optim.AdamW(self.model.parameters(),lr=self.conf['learn_rate'])
        self.sch =  torch.optim.lr_scheduler.StepLR(self.opti,step_size=5,last_epoch=-1)

    def _build_criterions(self): 
        weight_loss = self.conf['weight_loss']
        weights = None 
        if weight_loss and self.mode=='train': 
            weights = self.make_incidence_weights(self.loader_d['train'],device=self.device) 
        weights = None
        self.criterions = dict()
        self.criterions['class'] = CrossEntropyLoss(weight=weights)
    def make_incidence_weights(self,dl,device=None):
        df = dl.dataset.df.copy() 
        class_col = dl.dataset.task_col
        total = df.shape[0] 
        incidence= [ float(total/(df[class_col]==i).sum()) for i in sorted(df[class_col].unique()) ]
        return torch.tensor(incidence,device=device)

    def train_epoch(self):
        self.model = self.model.train()
        temp_o  = list() 
        temp_l = list() 
        for i,batch in tqdm(enumerate(self.loader_d['train']),total=len( self.loader_d['train'])): 
            inputs,labels,path = batch 
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterions['class'] (outputs,labels)
            self.opti.zero_grad() 
            loss.backward() 
            self.opti.step()
            self._log_scalar('batch_loss',loss.cpu().detach().item(),self.gb_step)
            self.gb_step  +=1 
        self.opti.zero_grad() 
        self.c_epoch +=1
    def val_epoch(self): 
        self.model = self.model.eval() 
        total_loss = 0 
        with torch.no_grad(): 
            for i,val_data in enumerate(self.loader_d['val']): 
                inputs,labels,path = val_data 
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterions['class'] (outputs,labels)
                total_loss += loss.cpu().item()
            self._log_scalar('val_loss',total_loss/len(self.loader_d['val']),self.c_epoch)
        return total_loss
    def test_model(self,split='ts'): 
        self.model = self.model.eval() 
        paths = list()
        preds = list() 
        with torch.no_grad(): 
            for i,val_data in tqdm(enumerate(self.loader_d[split]),total=len(self.loader_d[split])): 
                inputs,labels,path = val_data 
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs) 
                preds.append(softmax(outputs.cpu(),dim=1))
                paths.extend(path) 
        key_name= self.conf['col_info']['embed_col']
        pred_d = defaultdict(list)  
        for pred_row in preds: 
            for i in range(pred_row.shape[1]): 
                for j in range(pred_row.shape[0]):
                    pred_d[f'task_p_{i}'].append(pred_row[j,i].item())  
        pred_d[key_name] = paths
        pred_df = pd.DataFrame(pred_d)
        return pred_df    
    def infer_model(self,split='infer'):
        return self.test_model(split=split)        
    def fit(self): 
        num_epochs = self.total_epochs
        best_val_loss = 10000
        last_update = -1 
        for i in range(num_epochs):
            self.train_epoch()
            val_loss = self.val_epoch()
            self.sch.step(i) 
            if val_loss <= best_val_loss: 
                self.store_model()
                best_val_loss = val_loss 
                last_update = i 
            if (i-last_update) >=20: 
                print(f"Early STOP")
                break  
    def _fit_optuna(self,trial): 
        num_epochs = self.total_epochs
        best_val_loss = 10000
        last_update = -1 
        self.disable_bar = True
        for i in range(num_epochs):
            self.train_epoch()
            val_loss = self.val_epoch()
            self.sch.step(val_loss) 
            trial.report(val_loss,i)
            if trial.should_prune(): 
                raise optuna.exceptions.TrialPruned() 
        return val_loss
    @classmethod
    def get_trial_suggestions(cls,trial:optuna.Trial,c_conf):  
        lr = trial.suggest_float("learn_rate",0.0001,0.1) 
        decay = trial.suggest_float("weight_decay",0,0.1)
        c_conf['learn_rate'] = lr 
        c_conf['weight_decay'] = decay 
        return c_conf
    @classmethod 
    def _get_search_space(cls):
        return None 
    def store_model(self):
        model_dir = self.conf['log_dir']
        w_path = os.path.join(model_dir,'model_w.ckpt')
        torch.save({
            'conf':self.conf,
            'model_weights':self.model.state_dict(),
            'epoch':self.c_epoch,
            },f=w_path
        )
