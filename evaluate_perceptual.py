import torch
import torch.nn as nn 
import torch.nn.functional as F 
import argparse, os, random
import torch.optim as optimizer 
from optim import get_optim, adjust_lr
from torch.utils.data import DataLoader,Subset, dataloader,random_split
import numpy as np 
from tqdm import tqdm 
import time,datetime
import sys
sys.path.append("core/data_loader")

from config.config_model import *


from core.data_loader.load_data_t4sa import *
from core.data_loader.mosei_dataset import *
from core.data_loader.load_data_vsnli import *


from core.model.t4sa import T4sa
from core.model.t4sa_early import T4sa_Early
from core.model.mosei import Mosei
from core.model.mosei_early import Mosei_Early
from core.model.snli import SNLI
from core.model.snli_early import SNLI_Early 



def parse_args():
    parser = argparse.ArgumentParser()
    # Model

    ## Model and Dataset Selection
    parser.add_argument('--model', 
                        choices=['Dynamic', 'Dense', 'Late','Early'],
                        type=str,default="Dynamic")

    parser.add_argument('--dataset',
                        choices=['T4sa', 'Mosei', 'SNLI'],
                        type=str,default="T4sa")

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp_cofeature/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=99)
    parser.add_argument('--opt', type=str, default="Adam")
    parser.add_argument('--opt_params', type=str, default="{'betas': '(0.9, 0.98)', 'eps': '1e-9'}")
    parser.add_argument('--lr_base', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_times', type=int, default=3)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--ans_size', type=float, default=3)
    parser.add_argument('--pred_func',type=str, default="amax")
    # Dataset and task

    parser.add_argument('--average_number',type=int, default=1)
    parser.add_argument('--root_dir',type=str,default="/mnt/ssd/Datasets_/")
    parser.add_argument('--checkpoint',type=str,default="ckpt/")
    args = parser.parse_args()
    return args





def evaluate(net, eval_loader, args,device):
    """
    select_modal:   0:img  1:text 3:concat
    select_padding: 0:img  1:text 

    """

    accuracy = []
    net.train(False)

    for step,feat in enumerate(eval_loader):
        out,ans = net.feedforward(device,feat,net)
        out = out.cpu().data.numpy()
        ans = ans.cpu().data.numpy()
        accuracy += list(np.argmax(out, axis=1) == ans)
        
    net.train(True)
    return 100*np.mean(np.array(accuracy)) #, preds



def evaluate_perceptual(net, test_data_iter,args,device,average_number=1):
    acc_list = []
    for i in range(average_number):
        seed = np.random.choice([i for i in range(1000)],1)
        np.random.seed(seed)
        print("Calculation Start...")
        acc = evaluate(net, test_data_iter, args,device)
        acc_list.append(round(acc,2))
        print(round(acc,2))
    
    return np.mean(acc_list),np.std(acc_list)
    



def get_dataset_perceptual(config,select_modal,select_class):
    """
        if_train: True, False 
        get dataset for percetual socre calculation
        selct_padding: 0,1 --> x,y
        select_class:  0:in_class, 1:out_class
    """
    if config.dataset =="T4sa" and config.model!="Early":
        print("loading T4sa dataset...")
        train_dataset = T4SA_Vision_And_Text_CoFeature_Mask_Score_Perceptual(0,None,config,select_modal,select_class)  
        val_dataset =T4SA_Vision_And_Text_CoFeature_Mask_Score_Perceptual(1,train_dataset.token_to_ix,config,select_modal,select_class)
        test_dataset =T4SA_Vision_And_Text_CoFeature_Mask_Score_Perceptual(2,train_dataset.token_to_ix,config,select_modal,select_class)
        return train_dataset,val_dataset,test_dataset
    elif config.dataset=="T4sa" and config.model=="Early":
        train_dataset = T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion_Score_Perceptual(0,None,config,select_modal,select_class)    
        val_dataset =T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion_Score_Perceptual(1,train_dataset.token_to_ix,config,select_modal,select_class)
        test_dataset =T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion_Score_Perceptual(2,train_dataset.token_to_ix,config,select_modal,select_class)
        return train_dataset,val_dataset,test_dataset

    elif config.dataset=="Mosei" and config.model!="Early":
        print("loading Mosei dataset...")
        train_dataset = Mosei_Dataset_AT_Score_Perceptual('train',config,None,select_modal,select_class)  
        val_dataset =Mosei_Dataset_AT_Score_Perceptual('valid',config,train_dataset.token_to_ix,select_modal,select_class)
        test_dataset =Mosei_Dataset_AT_Score_Perceptual('test',config,train_dataset.token_to_ix,select_modal,select_class)
        return train_dataset,val_dataset,test_dataset
    
    elif config.dataset=="Mosei" and config.model=="Early":
        train_dataset = Mosei_Dataset_AT_Early_Fusion_Score_Perceptual('train',config,None,select_modal,select_class)  
        val_dataset = Mosei_Dataset_AT_Early_Fusion_Score_Perceptual('valid',config,train_dataset.token_to_ix,select_modal,select_class)
        test_dataset = Mosei_Dataset_AT_Early_Fusion_Score_Perceptual('test',config,train_dataset.token_to_ix,select_modal,select_class)
 
        return train_dataset,val_dataset,test_dataset

    elif config.dataset=="SNLI" and config.model!="Early":
        print("loading SNLI dataset...")
        train_dataset = VSNLI_Only_Text_Score_Perceptual(0,config,None,select_modal,select_class)  
        val_dataset =VSNLI_Only_Text_Score_Perceptual(1,config,train_dataset.token_to_ix,select_modal,select_class)
        test_dataset =VSNLI_Only_Text_Score_Perceptual(2,config,train_dataset.token_to_ix,select_modal,select_class)
        return train_dataset,val_dataset,test_dataset
    elif config.dataset=="SNLI" and config.model=="Early":
        print("loading SNLI early dataset...")
        train_dataset = VSNLI_Only_Text_Early_Score_Perceptual(0,config,None,select_modal,select_class)  
        val_dataset =VSNLI_Only_Text_Early_Score_Perceptual(1,config,train_dataset.token_to_ix,select_modal,select_class)
        test_dataset =VSNLI_Only_Text_Early_Score_Perceptual(2,config,train_dataset.token_to_ix,select_modal,select_class)
        return train_dataset,val_dataset,test_dataset
    
    else: 
        print("wrong dataset")



def get_model(config,train_dataset):
    if config.dataset =="T4sa" and config.model!="Early":
        net = T4sa(config, train_dataset.vocab_size, train_dataset.pretrained_emb)
        return net 
    elif config.dataset =="T4sa" and config.model=="Early": 
        net = T4sa_Early(config, train_dataset.vocab_size, train_dataset.pretrained_emb)
        return net  
    
    elif config.dataset =="Mosei" and config.model!="Early":
        net = Mosei(config, train_dataset.vocab_size, train_dataset.pretrained_emb)
        return net 
    elif config.dataset =="Mosei" and config.model=="Early": 
        net = Mosei_Early(config, train_dataset.vocab_size, train_dataset.pretrained_emb)
        return net  

    elif config.dataset =="SNLI" and config.model!="Early":
        net = SNLI(config, train_dataset.vocab_size, train_dataset.pretrained_emb)
        return net 

    elif config.dataset =="SNLI" and config.model=="Early": 
        net = SNLI_Early(config, train_dataset.vocab_size, train_dataset.pretrained_emb)
        return net
    

def run():

    #DataLoader_Seq50()
    args = parse_args()
    args_dict = vars(args) 


    result = []
    model_config = Config(args_dict)
    for modal in [0,1]:
        for class_type in [0,1]:
            select_modal = modal 
            select_class = class_type
            print("calc perceputal score...modal:{}, class:{}".format(select_modal,select_class))
            train_dataset,val_dataset,test_dataset = get_dataset_perceptual(model_config,select_modal,select_class)
            datasize = train_dataset.__len__()
    

            train_data_iter = DataLoader(train_dataset,batch_size=model_config.batch_size,shuffle=True,num_workers=32,pin_memory=True) #
            test_data_iter = DataLoader(test_dataset,batch_size=model_config.batch_size,shuffle=False,num_workers=32,pin_memory=True)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = get_model(model_config,train_dataset).to(device)
            net.load_state_dict(torch.load(model_config.checkpoint)['state_dict'])
            
            acc_mean,acc_std = evaluate_perceptual(net, test_data_iter,args,device,model_config.average_number)
            result.append([acc_mean,acc_std])
    
    
    print(result)

   

if __name__ == "__main__":
    run()
  
