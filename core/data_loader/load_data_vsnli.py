# --------------------------------------------------------
# Produce training dataset
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

import torch
import torch.utils.data.dataset as Data  
from torch.utils.data import DataLoader,Subset,Dataset
import torchvision.transforms as transforms
from tqdm import tqdm 
import numpy as np 
import pandas as pd 
import re
from PIL import Image
from util import * 
import time 




class VSNLI_Only_Text(Data.Dataset):
    """
    get premise and hyposis
    """

    def __init__(self,subset_choose,token_to_ix,config):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        """
        root_path= config.root_dir
        
        

        if subset_choose == 0:
            self.data = np.load(root_path+"vsnli/token/train.npy",allow_pickle=True).tolist()
        elif subset_choose ==1:
            self.data = np.load(root_path+"vsnli/token/dev.npy",allow_pickle=True).tolist()
        elif subset_choose ==2:
            self.data = np.load(root_path+"vsnli/token/test.npy",allow_pickle=True).tolist()
        else:
            self.data = np.load(root_path+"vsnli/token/test_hard.npy",allow_pickle=True).tolist()


        self.label = self.data[0]
        self.text_pre = self.data[1]
        self.text_asu = self.data[2]
        
        
        self.text_list = self.text_pre+self.text_asu
        
      

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        
        

    def __getitem__(self,idx):

        premise = self.text_pre[idx]
        hypothesis = self.text_asu[idx]

        
        text_pre = self.sent_to_ix_cls(premise,self.token_to_ix, max_token=50) # len:51 
        text_hyp = self.sent_to_ix_cls(hypothesis,self.token_to_ix,max_token=50) # len:51

        text_pre_mask = [True if i == 0 else False for i in text_pre]
        text_hyp_mask = [True if i == 0 else False for i in text_hyp]

        

        text_pre_mask = torch.tensor(text_pre_mask).unsqueeze(0).unsqueeze(1)
        
        text_hyp_mask = torch.tensor(text_hyp_mask).unsqueeze(0).unsqueeze(1)

        xy_mask = torch.cat((text_pre_mask,text_hyp_mask),dim=2)



        # print("face:",self.face_feat_list.__len__())
        ans_iter = self.label[idx]
       
        # image = Image.open(self.img[idx]).convert("RGB")
        # image = self.transform(image)
        
        return torch.from_numpy(text_pre),torch.from_numpy(text_hyp),text_pre_mask,text_hyp_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long) 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

    def sent_to_ix_cls(self,s, token_to_ix, max_token):
        ques_ix = np.zeros(max_token+1, np.int64)
        ques_ix[0] = token_to_ix['CLS']
        for ix, word in enumerate(s):
            #word = p.singular_noun(word)
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token+1:
                break

        return ques_ix

    def create_dict_list(self,sentence_list, use_glove=True):
    # token_file = dataroot+"/token_to_ix.pkl"
    # glove_file = dataroot+"/train_glove.npy"
    # if os.path.exists(glove_file) and os.path.exists(token_file):
    #     print("Loading train language files")
    #     return pickle.load(open(token_file, "rb")), np.load(glove_file)

        print("Creating train language files")
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'SEP': 2, 
            'CLS': 3,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('SEP').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)


        for v in sentence_list:
            for word in v:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        # np.save(glove_file, pretrained_emb)
        # pickle.dump(token_to_ix, open(token_file, "wb"))
        return token_to_ix, pretrained_emb




class VSNLI_Only_Text_Score_Perceptual(Data.Dataset):
    """
    get premise and hyposis
    """

    def __init__(self,subset_choose,config,token_to_ix=None,select_padding=0,select_class=0):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        """
        
        self.select_padding = select_padding
        self.select_class = select_class

        root_dir = config.root_dir
        if subset_choose == 0:
            self.data = np.load(root_dir+"vsnli/token/train.npy",allow_pickle=True).tolist()
        elif subset_choose ==1:
            self.data = np.load(root_dir+"vsnli/token/dev.npy",allow_pickle=True).tolist()
        elif subset_choose ==2:
            self.data = np.load(root_dir+"vsnli/token/test.npy",allow_pickle=True).tolist()
        else:
            self.data = np.load(root_dir+"vsnli/token/test_hard.npy",allow_pickle=True).tolist()


        self.label = self.data[0]
        self.text_pre = self.data[1]
        self.text_asu = self.data[2]
        
        
        self.text_list = self.text_pre+self.text_asu
        
      

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        self.class_0_label = np.load(root_dir+"vsnli/token/label_idx/test_label0.npy",allow_pickle=True).tolist()
        self.class_1_label = np.load(root_dir+"vsnli/token/label_idx/test_label1.npy",allow_pickle=True).tolist()
        self.class_2_label = np.load(root_dir+"vsnli/token/label_idx/test_label2.npy",allow_pickle=True).tolist()
      
        
        

    def __getitem__(self,idx):
        ans_iter = self.label[idx]

        labe_list = self.class_0_label+self.class_1_label+self.class_2_label
        new_key = np.random.choice(labe_list)
        
            
    
        if self.select_padding ==0:
            premise = self.text_pre[new_key]
            hypothesis = self.text_asu[idx]

            
            text_pre = self.sent_to_ix_cls(premise,self.token_to_ix, max_token=50) # len:51 
            text_hyp = self.sent_to_ix_cls(hypothesis,self.token_to_ix,max_token=50) # len:51

            text_pre_mask = [True if i == 0 else False for i in text_pre]
            text_hyp_mask = [True if i == 0 else False for i in text_hyp]

            

            text_pre_mask = torch.tensor(text_pre_mask).unsqueeze(0).unsqueeze(1)
            
            text_hyp_mask = torch.tensor(text_hyp_mask).unsqueeze(0).unsqueeze(1)

            xy_mask = torch.cat((text_pre_mask,text_hyp_mask),dim=2)

        else:
            premise = self.text_pre[idx]
            hypothesis = self.text_asu[new_key]

            
            text_pre = self.sent_to_ix_cls(premise,self.token_to_ix, max_token=50) # len:51 
            text_hyp = self.sent_to_ix_cls(hypothesis,self.token_to_ix,max_token=50) # len:51

            text_pre_mask = [True if i == 0 else False for i in text_pre]
            text_hyp_mask = [True if i == 0 else False for i in text_hyp]

            

            text_pre_mask = torch.tensor(text_pre_mask).unsqueeze(0).unsqueeze(1)
            
            text_hyp_mask = torch.tensor(text_hyp_mask).unsqueeze(0).unsqueeze(1)

            xy_mask = torch.cat((text_pre_mask,text_hyp_mask),dim=2)


        # print("face:",self.face_feat_list.__len__())
        
       
        # image = Image.open(self.img[idx]).convert("RGB")
        # image = self.transform(image)
        
        return torch.from_numpy(text_pre),torch.from_numpy(text_hyp),text_pre_mask,text_hyp_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long) 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

    def sent_to_ix_cls(self,s, token_to_ix, max_token):
        ques_ix = np.zeros(max_token+1, np.int64)
        ques_ix[0] = token_to_ix['CLS']
        for ix, word in enumerate(s):
            #word = p.singular_noun(word)
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token+1:
                break

        return ques_ix

    def create_dict_list(self,sentence_list, use_glove=True):
    # token_file = dataroot+"/token_to_ix.pkl"
    # glove_file = dataroot+"/train_glove.npy"
    # if os.path.exists(glove_file) and os.path.exists(token_file):
    #     print("Loading train language files")
    #     return pickle.load(open(token_file, "rb")), np.load(glove_file)

        print("Creating train language files")
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'SEP': 2, 
            'CLS': 3,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('SEP').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)


        for v in sentence_list:
            for word in v:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        # np.save(glove_file, pretrained_emb)
        # pickle.dump(token_to_ix, open(token_file, "wb"))
        return token_to_ix, pretrained_emb


class VSNLI_Only_Text_Early(Data.Dataset):
    """
    get premise and hyposis
    """

    def __init__(self,subset_choose,token_to_ix,config):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        """
        
        # if subset_choose == 0:
        #     self.data = pd.read_csv("/mnt/ssd/Datasets_/vsnli/vsnli/VSNLI_1.0_train.tsv",sep='\t')
        #     # delete nan items 
        #     self.data=self.data.drop([90649,90650,90651,308489,308490,308491],axis=0)

        # elif subset_choose == 1:
        #     self.data = pd.read_csv("/mnt/ssd/Datasets_/vsnli/vsnli/VSNLI_1.0_dev.tsv",sep='\t')
        # elif subset_choose==2:
        #     self.data = pd.read_csv("/mnt/ssd/Datasets_/vsnli/vsnli/VSNLI_1.0_test.tsv",sep='\t')
        # else: 
        #     self.data = pd.read_csv("/mnt/ssd/Datasets_/vsnli/vsnli/VSNLI_1.0_test_hard.tsv",sep='\t')
        root_path= config.root_dir

        if subset_choose == 0:
            self.data = np.load(root_path+"vsnli/token/train.npy",allow_pickle=True).tolist()
        elif subset_choose ==1:
            self.data = np.load(root_path+"vsnli/token/dev.npy",allow_pickle=True).tolist()
        elif subset_choose ==2:
            self.data = np.load(root_path+"vsnli/token/test.npy",allow_pickle=True).tolist()
        else:
            self.data = np.load(root_path+"vsnli/token/test_hard.npy",allow_pickle=True).tolist()


        self.label = self.data[0]
        self.text_pre = self.data[1]
        self.text_asu = self.data[2]
        
        
        self.text_list = self.text_pre+self.text_asu
        
      

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        
        

    def __getitem__(self,idx):

        premise = self.text_pre[idx]
        hypothesis = self.text_asu[idx]

        
        text_pre = self.sent_to_ix_cls(premise,self.token_to_ix, max_token=50) # len:51 
        text_hyp = self.sent_to_ix(hypothesis,self.token_to_ix,max_token=50) # len:51

        text_pre_mask = [True if i == 0 else False for i in text_pre]
        text_hyp_mask = [True if i == 0 else False for i in text_hyp]

        

        text_pre_mask = torch.tensor(text_pre_mask).unsqueeze(0).unsqueeze(1)
        
        text_hyp_mask = torch.tensor(text_hyp_mask).unsqueeze(0).unsqueeze(1)

        xy_mask = torch.cat((text_pre_mask,text_hyp_mask),dim=2)



        # print("face:",self.face_feat_list.__len__())
        ans_iter = self.label[idx]
       
        # image = Image.open(self.img[idx]).convert("RGB")
        # image = self.transform(image)
        
        return torch.from_numpy(text_pre),torch.from_numpy(text_hyp),xy_mask,torch.tensor(ans_iter,dtype=torch.long) 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

    def sent_to_ix_cls(self,s, token_to_ix, max_token):
        ques_ix = np.zeros(max_token+1, np.int64)
        ques_ix[0] = token_to_ix['CLS']
        for ix, word in enumerate(s):
            #word = p.singular_noun(word)
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token+1:
                break

        return ques_ix

    def sent_to_ix(self,s, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
       
        for ix, word in enumerate(s):
            #word = p.singular_noun(word)
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def create_dict_list(self,sentence_list, use_glove=True):
    # token_file = dataroot+"/token_to_ix.pkl"
    # glove_file = dataroot+"/train_glove.npy"
    # if os.path.exists(glove_file) and os.path.exists(token_file):
    #     print("Loading train language files")
    #     return pickle.load(open(token_file, "rb")), np.load(glove_file)

        print("Creating train language files")
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'SEP': 2, 
            'CLS': 3,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('SEP').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)


        for v in sentence_list:
            for word in v:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        # np.save(glove_file, pretrained_emb)
        # pickle.dump(token_to_ix, open(token_file, "wb"))
        return token_to_ix, pretrained_emb





class VSNLI_Only_Text_Early_Score_Perceptual(Data.Dataset):
    """
    get premise and hyposis
    """

    def __init__(self,subset_choose,config,token_to_ix,select_padding=0,select_class=0):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        """
        self.select_padding = select_padding
        self.select_class = select_class
       
        root_dir = config.root_dir 
        
        if subset_choose == 0:
            self.data = np.load(root_dir+"vsnli/token/train.npy",allow_pickle=True).tolist()
        elif subset_choose ==1:
            self.data = np.load(root_dir+"vsnli/token/dev.npy",allow_pickle=True).tolist()
        elif subset_choose ==2:
            self.data = np.load(root_dir+"vsnli/token/test.npy",allow_pickle=True).tolist()
        else:
            self.data = np.load(root_dir+"vsnli/token/test_hard.npy",allow_pickle=True).tolist()


        self.label = self.data[0]
        self.text_pre = self.data[1]
        self.text_asu = self.data[2]
        
        
        self.text_list = self.text_pre+self.text_asu
        
      

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        self.class_0_label = np.load(root_dir+"vsnli/token/label_idx/test_label0.npy",allow_pickle=True).tolist()
        self.class_1_label = np.load(root_dir+"vsnli/token/label_idx/test_label1.npy",allow_pickle=True).tolist()
        self.class_2_label = np.load(root_dir+"vsnli/token/label_idx/test_label2.npy",allow_pickle=True).tolist()
        

    def __getitem__(self,idx):
        ans_iter = self.label[idx]

        labe_list = self.class_0_label+self.class_1_label+self.class_2_label
        new_key = np.random.choice(labe_list)
        
        if self.select_padding ==0:
            premise = self.text_pre[new_key]
            hypothesis = self.text_asu[idx]

            
            text_pre = self.sent_to_ix_cls(premise,self.token_to_ix, max_token=50) # len:51 
            text_hyp = self.sent_to_ix(hypothesis,self.token_to_ix,max_token=50) # len:51

            text_pre_mask = [True if i == 0 else False for i in text_pre]
            text_hyp_mask = [True if i == 0 else False for i in text_hyp]

            

            text_pre_mask = torch.tensor(text_pre_mask).unsqueeze(0).unsqueeze(1)
            
            text_hyp_mask = torch.tensor(text_hyp_mask).unsqueeze(0).unsqueeze(1)

            xy_mask = torch.cat((text_pre_mask,text_hyp_mask),dim=2)
        else:
            premise = self.text_pre[idx]
            hypothesis = self.text_asu[new_key]

            
            text_pre = self.sent_to_ix_cls(premise,self.token_to_ix, max_token=50) # len:51 
            text_hyp = self.sent_to_ix(hypothesis,self.token_to_ix,max_token=50) # len:51

            text_pre_mask = [True if i == 0 else False for i in text_pre]
            text_hyp_mask = [True if i == 0 else False for i in text_hyp]

            

            text_pre_mask = torch.tensor(text_pre_mask).unsqueeze(0).unsqueeze(1)
            
            text_hyp_mask = torch.tensor(text_hyp_mask).unsqueeze(0).unsqueeze(1)

            xy_mask = torch.cat((text_pre_mask,text_hyp_mask),dim=2)



        # print("face:",self.face_feat_list.__len__())
        
       
        # image = Image.open(self.img[idx]).convert("RGB")
        # image = self.transform(image)
        
        return torch.from_numpy(text_pre),torch.from_numpy(text_hyp),xy_mask,torch.tensor(ans_iter,dtype=torch.long) 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

    def sent_to_ix_cls(self,s, token_to_ix, max_token):
        ques_ix = np.zeros(max_token+1, np.int64)
        ques_ix[0] = token_to_ix['CLS']
        for ix, word in enumerate(s):
            #word = p.singular_noun(word)
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token+1:
                break

        return ques_ix

    def sent_to_ix(self,s, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
       
        for ix, word in enumerate(s):
            #word = p.singular_noun(word)
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def create_dict_list(self,sentence_list, use_glove=True):
    # token_file = dataroot+"/token_to_ix.pkl"
    # glove_file = dataroot+"/train_glove.npy"
    # if os.path.exists(glove_file) and os.path.exists(token_file):
    #     print("Loading train language files")
    #     return pickle.load(open(token_file, "rb")), np.load(glove_file)

        print("Creating train language files")
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'SEP': 2, 
            'CLS': 3,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('SEP').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)


        for v in sentence_list:
            for word in v:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        # np.save(glove_file, pretrained_emb)
        # pickle.dump(token_to_ix, open(token_file, "wb"))
        return token_to_ix, pretrained_emb






if __name__=='__main__':
    pass 
    # data = VSNLI_Vision_And_Text(0,None) 
    
    # train_data_iter = DataLoader(data,batch_size=1,shuffle=True,num_workers=4) #
    # for item in tqdm(train_data_iter):
    #     print(item[1].shape) 
    #     print(item)
    #     print(list(data.token_to_ix.keys())[0:10])
    #     print(list(data.token_to_ix.values())[0:10])
    #     break 
       
    
   