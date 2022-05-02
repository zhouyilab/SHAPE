import torch
import torch.nn as nn 
import torch.nn.functional as F 
import argparse, os, random
import torch.optim as optimizer 
from optim import get_optim, adjust_lr
from torch.utils.data import DataLoader,Subset, dataloader,random_split
import numpy as np 
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

    parser.add_argument('--root_dir',type=str,default="/mnt/ssd/Datasets_/")
    args = parser.parse_args()
    return args



def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 







def train(net, train_loader, eval_loader, args,device,datasize):

    path = args.output + "/" + args.name

   
    if not os.path.exists(path):
        os.mkdir(path)


    logfile = open(path+'/log_run.txt','a+')
    logfile.write(str(args))

    logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
    logfile.close()



    loss_sum = 0
    best_eval_accuracy = 0.0
    early_stop = 0
    decay_count = 0

    # Load the optimizer paramters
    #optim = torch.optim.Adam(net.parameters(), lr=args.lr_base)
    optim = get_optim(args,net,datasize)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    eval_accuracies = []
    for epoch in range(0, args.max_epoch):

        time_start = time.time()
        for step,feat in enumerate(train_loader):
            loss_tmp=0
            optim.zero_grad()
            out,ans = net.feedforward(device,feat,net)

            loss = loss_fn(out, ans)
            loss.backward()

            loss_sum += loss.cpu().data.numpy()
            loss_tmp += loss.cpu().data.numpy()

            print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m "
                  "remaining" % (
                      epoch + 1,
                      step,
                      int(len(train_loader.dataset) / args.batch_size),
                      loss_tmp / args.batch_size,
                      *[group['lr'] for group in optim.optimizer.param_groups],
                      ((time.time() - time_start) / (step + 1)) * ((len(train_loader.dataset) / args.batch_size) - step) / 60,
                  ), end='          ')

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    args.grad_norm_clip
                )

            optim.step()

        time_end = time.time()
        elapse_time = time_end-time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Logging

        logfile = open(path+'/log_run.txt','a+')
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / len(train_loader.dataset)) +
            ', Lr: ' + str([group['lr'] for group in optim.optimizer.param_groups]) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) +
            ', Speed(s/batch): ' + str(elapse_time / step)
        )
        logfile.close()

        # Eval
        if epoch_finish >= args.eval_start:
            print('Evaluation...')
            accuracy = evaluate(net, eval_loader, args,device)
            print('Accuracy :'+str(accuracy))

            logfile = open(path+'/log_run.txt','a+')
            logfile.write(
                'Accuracy: ' + str(accuracy) +
                '\n\n'
            )
            logfile.close()


            eval_accuracies.append(accuracy)
            if accuracy > best_eval_accuracy:
                # Best
                state = {
                    'state_dict': net.state_dict(),
                    'optimizer': optim.optimizer.state_dict(),
                    'args': args,
                }
                torch.save(
                    state,
                    args.output + "/" + args.name +
                    '/best'+str(args.seed)+'.pkl'
                )
                best_eval_accuracy = accuracy
                early_stop = 0

            elif decay_count < args.lr_decay_times:
                # Decay
                print('LR Decay...')
                decay_count += 1
                net.load_state_dict(torch.load(args.output + "/" + args.name +
                                               '/best'+str(args.seed)+'.pkl')['state_dict'])
                adjust_lr(optim, args.lr_decay)
                # for group in optim.optimizer.param_groups:
                #     group['lr'] *= args.lr_decay

            else:
                # Early stop
                early_stop += 1
                if early_stop == args.early_stop:
                    logfile = open(path+'/log_run.txt','a+')
                    logfile.write('Early stop reached' + '\n')
                    print('Early stop reached')
                    logfile.write('best_overall_acc :' + str(best_eval_accuracy) + '\n\n')
                    print('best_eval_acc :' + str(best_eval_accuracy) + '\n\n')
                    os.rename(args.output + "/" + args.name +
                              '/best'+str(args.seed)+'.pkl',
                              args.output + "/" + args.name +
                              '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + '.pkl')
                    logfile.close()
                    return eval_accuracies

        loss_sum = 0






def evaluate(net, eval_loader, args,device):
    accuracy = []
    net.train(False)
    preds = []
    for step,feat in enumerate(eval_loader):
        out,ans = net.feedforward(device,feat,net)
        out = out.cpu().data.numpy()
        ans = ans.cpu().data.numpy()
        accuracy += list(np.argmax(out, axis=1) == ans)
        
    net.train(True)
    return 100*np.mean(np.array(accuracy)) #, preds




def get_dataset(config,if_train,train_dataset):
    """
        if_train: True, False 
    """
    if config.dataset =="T4sa" and config.model!="Early":
        print("loading T4sa dataset...")
        if if_train:
            train_dataset = T4SA_Vision_And_Text_CoFeature_Mask(0,None,config)  
            val_dataset =T4SA_Vision_And_Text_CoFeature_Mask(1,train_dataset.token_to_ix,config)
            return train_dataset,val_dataset
        else: 
            test_dataset =T4SA_Vision_And_Text_CoFeature_Mask(2,train_dataset.token_to_ix,config)
            return test_dataset
    elif config.dataset=="T4sa" and config.model=="Early":
        print("loading T4sa early dataset...")
        if if_train:
            train_dataset = T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion(0,None,config)  
            val_dataset =T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion(1,train_dataset.token_to_ix,config) 
            return train_dataset,val_dataset
        else: 
            test_dataset =T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion(2,train_dataset.token_to_ix,config)
            return test_dataset  

    elif config.dataset=="Mosei" and config.model!="Early":
        print("loading Mosei dataset...")
        if if_train:
            train_dataset = Mosei_Dataset_AT('train',config,None)  
            val_dataset =Mosei_Dataset_AT('valid',config,train_dataset.token_to_ix)
            return train_dataset,val_dataset
        else: 
            test_dataset =Mosei_Dataset_AT('test',config,train_dataset.token_to_ix)
            return test_dataset

    elif config.dataset=="Mosei" and config.model=="Early":
        print("loading Mosei early dataset...")
        if if_train:
            train_dataset = Mosei_Dataset_AT_Early_Fusion('train',config,None)  
            val_dataset =Mosei_Dataset_AT_Early_Fusion('valid',config,train_dataset.token_to_ix)
            return train_dataset,val_dataset
        else: 
            test_dataset =Mosei_Dataset_AT_Early_Fusion('test',config,train_dataset.token_to_ix)
            return test_dataset

    elif config.dataset=="SNLI" and config.model!="Early":
        print("loading SNLI dataset...")
        if if_train:
            train_dataset = VSNLI_Only_Text(0,None,config)  
            val_dataset =VSNLI_Only_Text(1,train_dataset.token_to_ix,config)
            return train_dataset,val_dataset
        else: 
            test_dataset =VSNLI_Only_Text(2,train_dataset.token_to_ix,config)
            return test_dataset

    elif config.dataset=="SNLI" and config.model=="Early":
        print("loading SNLI early dataset...")
        if if_train:
            train_dataset = VSNLI_Only_Text_Early(0,None,config)  
            val_dataset =VSNLI_Only_Text_Early(1,train_dataset.token_to_ix,config)
            return train_dataset,val_dataset
        else: 
            test_dataset =VSNLI_Only_Text_Early(2,train_dataset.token_to_ix,config)
            return test_dataset
    
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

    model_config = Config(args_dict)




    train_dataset,val_dataset = get_dataset(model_config,True,None)
    datasize = train_dataset.__len__()
    train_data_iter = DataLoader(train_dataset,batch_size=model_config.batch_size,shuffle=True,num_workers=32,pin_memory=True) #
    val_data_iter = DataLoader(val_dataset,batch_size=model_config.batch_size,shuffle=False,num_workers=32,pin_memory=True)
    

    device = torch.device('cuda:1')
    net = get_model(model_config,train_dataset)
    net.to(device)
   

    train(net, train_data_iter, val_data_iter, model_config,device,datasize)



if __name__ == "__main__":
    run()


