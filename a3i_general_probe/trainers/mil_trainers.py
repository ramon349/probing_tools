from torch.nn import  CrossEntropyLoss
from torch import optim
from .trainer_factory import TrainerRegister 
from tqdm import tqdm 
import torch 
import os 
from torch.nn.functional import softmax
from collections import defaultdict
import pandas as pd 
from torchsurv.loss.cox import neg_partial_log_likelihood
from sklearn.metrics import recall_score,precision_score,roc_auc_score
from collections import deque
from torch.nn import BCELoss
from flash.core.optimizers import LinearWarmupCosineAnnealingLR 
import optuna 
import numpy as np 
def make_incidence_weights(dl,device=None):
    df = dl.dataset.df.copy() 
    class_col = dl.dataset.label_col
    total = df.shape[0] 
    incidence= [ float(total/(df[class_col]==i).sum()) for i in sorted(df[class_col].unique()) ]
    return torch.tensor(incidence,device=device)
@TrainerRegister.register(cls_name="attn_mil")
class MILTrainer:
    def __init__(self,model=None,conf=None,loader_dict=None,tb_writter=None):
        self.conf = conf
        self.model = model 
        self.tb: SummaryWriter =   tb_writter
        self.total_epochs = conf['epochs']
        self.loader_d = loader_dict
        self.device = conf['device']
        self._log_model_graph()
        self._build_criterions()
        self.init_optims()
        self.gb_step = 0 
        self.c_epoch =0
        self.disable_bar =False  
    def _log_model_graph(self):
        sample = next(iter(self.loader_d['train']))[0][0]
        self.tb.add_graph(self.model,sample.to(self.device)) 
    def _build_criterions(self): 
        weight_loss = self.conf['weight_loss']
        weights = None 
        if weight_loss: 
            weights = make_incidence_weights(self.loader_d['train'],device=self.device) 
            print(f"We have weights: {weights}")
        weights = None
        self.criterions = dict()
        self.criterions['class'] = CrossEntropyLoss(weight=weights)
    
    def init_optims(self): 
        self.opti = optim.SGD(self.model.parameters(),lr=self.conf['learn_rate'],weight_decay=self.conf['weight_decay'])
        self.sch =  LinearWarmupCosineAnnealingLR(self.opti,warmup_epochs=1,max_epochs=self.total_epochs,warmup_start_lr=0.00001)
    def _log_scalar(self,name,val,step): 
        if self.tb: 
            self.tb.add_scalar(name,val,global_step=step)
    def train_epoch(self):
        self.model = self.model.train()
        temp_o  = list() 
        temp_l = list() 
        for i,batch in tqdm(enumerate(self.loader_d['train']),total=len( self.loader_d['train'])): 
            inputs,labels,path = batch 
            inputs = inputs[0].to(self.device)
            labels = labels.to(self.device)
            _,_,outputs = self.model(inputs)
            loss = self.criterions['class'] (outputs,labels)
            loss.backward() 
            self._log_scalar('batch_loss',loss.cpu().detach().item(),self.gb_step)
            self.gb_step  +=1 
            temp_o  = list() 
            temp_l = list() 
            self.opti.step()
            self.opti.zero_grad() 
        self.opti.zero_grad() 
        self.c_epoch +=1
    def val_epoch(self): 
        self.model = self.model.eval() 
        total_loss = 0 
        all_preds = list() 
        all_gts  = list() 
        with torch.no_grad(): 
            for i,val_data in enumerate(self.loader_d['val']): 
                inputs,labels,path = val_data 
                inputs = inputs[0].to(self.device)
                labels = labels.to(self.device)
                _,_,outputs = self.model(inputs)
                loss = self.criterions['class'] (outputs,labels)
                total_loss += loss.cpu().item()
                all_preds.append(outputs.cpu())
                all_gts.append(labels.cpu())
            #self.tb.add_scalar('val_loss',total_loss/len(self.loader_d['val']),global_step=self.c_epoch) 
            self._log_scalar('val_loss',total_loss/len(self.loader_d['val']),self.c_epoch)
        return total_loss
    def test_model(self): 
        self.model = self.model.eval() 
        total_loss = 0 
        paths = list()
        preds = list() 
        with torch.no_grad(): 
            for i,val_data in tqdm(enumerate(self.loader_d['ts']),total=len(self.loader_d['ts'])): 
                inputs,labels,path = val_data 
                inputs = inputs[0].to(self.device)
                labels = labels.to(self.device)
                _,_,outputs = self.model(inputs) 
                preds.append(softmax(outputs.cpu()))
                paths.extend(path) 
        key_name= self.conf['feat_col']
        num_cols = preds[0].shape[1] 
        pred_d = defaultdict(list)  
        for pred_row in preds: 
            for i in range(pred_row.shape[1]): 
                pred_d[f'pred_{i}'].append(pred_row[0,i].item())  
        pred_d[key_name] = paths
        pred_df = pd.DataFrame(pred_d)
        return pred_df 
    def calc_perf_metrics(self,preds,labels):
        num_classes =  preds.shape[1]
        preds = softmax(preds,dim=1)
        max_preds = preds.argmax(dim=1)
        perf_dict= dict()
        bin_metrics = {'precision':precision_score,'recall':recall_score}
        for i in range(num_classes):
            for metric,func in bin_metrics.items():
                perf_dict[f'{i}_{metric}'] = func(labels==i,max_preds==i)
            perf_dict[f'val_{i}_auc']= roc_auc_score(labels==i,preds[:,i])
        return perf_dict
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
    
    def store_model(self):
        model_dir = self.conf['log_dir']
        w_path = os.path.join(model_dir,'model_w.ckpt')
        torch.save({
            'conf':self.conf,
            'model_weights':self.model.state_dict(),
            'epoch':self.c_epoch,
            },f=w_path
        )
    def test_model(self,split='ts'): 
        self.model = self.model.eval() 
        paths = list()
        preds = list() 
        with torch.no_grad(): 
            for i,val_data in tqdm(enumerate(self.loader_d[split]),total=len(self.loader_d[split])): 
                inputs,labels,path = val_data 
                inputs = inputs[0].to(self.device)
                labels = labels.to(self.device)
                _,_,outputs = self.model(inputs) 
                preds.append(softmax(outputs.cpu()))
                paths.extend(path) 
        key_name= self.conf['col_info']['feat_col']
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
