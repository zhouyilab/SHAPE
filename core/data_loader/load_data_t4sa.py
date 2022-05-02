from numpy.lib.function_base import select
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
import cv2 





class T4SA_Vision_And_Text_CoFeature_Mask(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix,args):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        padding version : img_len = text_len 
        """
        self.args = args 
        if subset_choose == 0:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/test.npy")
        


        
        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        

        if subset_choose == 0:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/train_all.npy")
        if subset_choose ==1:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/valid_all.npy")
        if subset_choose ==2:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/test_all.npy")
        

        self.transform =  transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        
        

    def __getitem__(self,idx):

        
        text = self.data_text[idx]
        text = self.sent_to_ix_cls(text,self.token_to_ix, max_token=self.args.text_len-1)
        
        text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

        xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
        
        xy_mask = torch.cat((xy_mask,text_mask),dim=-1)



        # print("face:",self.face_feat_list.__len__())
        ans_iter = int(self.label[idx])
       
        # image = Image.open(self.img[idx]).convert("RGB")
        # image = self.transform(image)
        



        path = self.args.root_dir+"t4sa/"+self.data_img[idx][1]
        img = cv2.imread(path)
        #img = img[:,:,(2,1,0)]
        #img = jpeg.JPEG(path).decode()
        img = self.transform(img)
        img =img.squeeze(0)
       
        
        # 49,2048
        return img,torch.from_numpy(text),text_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long)
 

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
                ques_ix[ix+1] = token_to_ix[word]
            else:
                ques_ix[ix+1] = token_to_ix['UNK']

            if ix + 2 == max_token+1:
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


class T4SA_Vision_And_Text_CoFeature_Mask_Score(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix,args,select_padding,select_class):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        padding version : img_len = text_len 
        select_padding: 0:img, 1:text 
        select_class:   0:in_class, 1:out_class 
        """
        self.select_padding = select_padding
        self.select_class = select_class 

        self.args = args 
        if subset_choose == 0:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/test.npy")
        

        self.class0_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label0.npy").tolist()
        self.class1_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label1.npy").tolist()
        self.class2_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label2.npy").tolist()

        
        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        

        if subset_choose == 0:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/train_all.npy")
        if subset_choose ==1:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/valid_all.npy")
        if subset_choose ==2:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/test_all.npy")
        

        self.transform =  transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        length = self.label.__len__()
        

    def __getitem__(self,idx):
        
        ans_iter = int(self.label[idx])

        # select_padding: 0:img, 1:text 
        new_id = 0 
            # select_class:   0:in_class, 1:out_class 
        if ans_iter == 0:
            if self.select_class == 0:
                new_id = np.random.choice(self.class0_idx, size=1, replace=False)[0] 
            else:
                new_id = np.random.choice(self.class1_idx+self.class2_idx,size=1,replace=False)[0]
        if ans_iter == 1:
            if self.select_class == 0:
                new_id = np.random.choice(self.class1_idx, size=1, replace=False)[0] 
            else:
                new_id = np.random.choice(self.class0_idx+self.class2_idx,size=1,replace=False)[0]

        if ans_iter == 2:
            if self.select_class == 0:
                new_id = np.random.choice(self.class2_idx, size=1, replace=False)[0] 
            else:
                new_id = np.random.choice(self.class0_idx+self.class1_idx,size=1,replace=False)[0]

        if self.select_padding == 0:
            path = self.args.root_dir+"t4sa/"+self.data_img[new_id][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
    
            text = self.data_text[idx]
            text = self.sent_to_ix_cls(text,self.token_to_ix, max_token=self.args.text_len-1)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)
        else: 
            path = self.args.root_dir+"t4sa/"+self.data_img[idx][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
    
            text = self.data_text[new_id]
            text = self.sent_to_ix_cls(text,self.token_to_ix, max_token=self.args.text_len-1)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)

        
        # 49,2048
        return img,torch.from_numpy(text),text_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long)
 

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
                ques_ix[ix+1] = token_to_ix[word]
            else:
                ques_ix[ix+1] = token_to_ix['UNK']

            if ix + 2 == max_token+1:
                break

        return ques_ix

    def create_dict_list(self,sentence_list, use_glove=True):

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


class T4SA_Vision_And_Text_CoFeature_Mask_Score_Perceptual(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix,args,select_padding,select_class):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        padding version : img_len = text_len 
        select_padding: 0:img, 1:text 
        select_class:   0:in_class, 1:out_class 
        """
        self.select_padding = select_padding
        self.select_class = select_class 

        self.args = args 
        if subset_choose == 0:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/test.npy")
        

        self.class0_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label0.npy").tolist()
        self.class1_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label1.npy").tolist()
        self.class2_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label2.npy").tolist()

        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        

        if subset_choose == 0:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/train_all.npy")
        if subset_choose ==1:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/valid_all.npy")
        if subset_choose ==2:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/test_all.npy")
        

        self.transform =  transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        length = self.label.__len__()
        

    def __getitem__(self,idx):
        
        ans_iter = int(self.label[idx])

        # select_padding: 0:img, 1:text 
        
            # select_class:   0:in_class, 1:out_class 
        label_list = self.class0_idx+self.class1_idx+self.class2_idx
        new_id = np.random.choice(label_list, size=1, replace=True)[0]


        if self.select_padding == 0:
            path = self.args.root_dir+"t4sa/"+self.data_img[new_id][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
    
            text = self.data_text[idx]
            text = self.sent_to_ix_cls(text,self.token_to_ix, max_token=self.args.text_len-1)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)
        else: 
            path = self.args.root_dir+"t4sa/"+self.data_img[idx][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
    
            text = self.data_text[new_id]
            text = self.sent_to_ix_cls(text,self.token_to_ix, max_token=self.args.text_len-1)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)

        
        # 49,2048
        return img,torch.from_numpy(text),text_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long)
 

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
                ques_ix[ix+1] = token_to_ix[word]
            else:
                ques_ix[ix+1] = token_to_ix['UNK']

            if ix + 2 == max_token+1:
                break

        return ques_ix

    def create_dict_list(self,sentence_list, use_glove=True):


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

class T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix,args):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        padding version : img_len = text_len 
        """
        self.args = args 
        if subset_choose == 0:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/test.npy")
        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()


        if subset_choose == 0:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/train_all.npy")
        if subset_choose ==1:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/valid_all.npy")
        if subset_choose ==2:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/test_all.npy")
        

        self.transform =  transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        
        

    def __getitem__(self,idx):

        
        text = self.data_text[idx]
        text = self.sent_to_ix(text,self.token_to_ix, max_token=self.args.text_len)
        
        text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

        xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
        
        xy_mask = torch.cat((xy_mask,text_mask),dim=-1)


        # print("face:",self.face_feat_list.__len__())
        ans_iter = int(self.label[idx])
    
        path = self.args.root_dir+"t4sa/"+self.data_img[idx][1]
        img = cv2.imread(path)
        # img = img[:,:,(2,1,0)]
        img = self.transform(img)
        img =img.squeeze(0)
       
        
        # 49,2048
        return img,torch.from_numpy(text),text_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long)
 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

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



class T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion_Score(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix,args,select_padding,select_class):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        padding version : img_len = text_len 
        """
        self.select_padding = select_padding
        self.select_class = select_class 
        self.args = args 
        if subset_choose == 0:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/test.npy")
        


        self.class0_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label0.npy").tolist()
        self.class1_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label1.npy").tolist()
        self.class2_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label2.npy").tolist()
        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        

        if subset_choose == 0:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/train_all.npy")
        if subset_choose ==1:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/valid_all.npy")
        if subset_choose ==2:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/test_all.npy")
        

        self.transform =  transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        
        

    def __getitem__(self,idx):
        ans_iter = int(self.label[idx])

         
            # select_class:   0:in_class, 1:out_class 
        if ans_iter == 0:
            if self.select_class == 0:
                new_id = np.random.choice(self.class0_idx, size=1, replace=False)[0] 
            else:
                new_id = np.random.choice(self.class1_idx+self.class2_idx,size=1,replace=False)[0]
        if ans_iter == 1:
            if self.select_class == 0:
                new_id = np.random.choice(self.class1_idx, size=1, replace=False)[0] 
            else:
                new_id = np.random.choice(self.class0_idx+self.class2_idx,size=1,replace=False)[0]

        if ans_iter == 2:
            if self.select_class == 0:
                new_id = np.random.choice(self.class2_idx, size=1, replace=False)[0] 
            else:
                new_id = np.random.choice(self.class0_idx+self.class1_idx,size=1,replace=False)[0]
        
        if self.select_padding == 0:
            text = self.data_text[idx]
            text = self.sent_to_ix(text,self.token_to_ix, max_token=self.args.text_len)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)




            path = self.args.root_dir+"t4sa/"+self.data_img[new_id][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
        else:
            text = self.data_text[new_id]
            text = self.sent_to_ix(text,self.token_to_ix, max_token=self.args.text_len)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)





            path = self.args.root_dir+"t4sa/"+self.data_img[idx][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
            
        # 49,2048
        return img,torch.from_numpy(text),text_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long)
 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

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



class T4SA_Vision_And_Text_CoFeature_Mask_EarlyFusion_Score_Perceptual(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix,args,select_padding,select_class):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        padding version : img_len = text_len 
        """
        self.select_padding = select_padding
        self.select_class = select_class 
        self.args = args 
        if subset_choose == 0:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load(self.args.root_dir+"t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load(self.args.root_dir+"t4sa/data_pro/labels/test.npy")
        


        self.class0_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label0.npy").tolist()
        self.class1_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label1.npy").tolist()
        self.class2_idx = np.load(self.args.root_dir+"t4sa/data_pro/test_label_idx/label2.npy").tolist()
        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        
      
        

        if subset_choose == 0:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/train_all.npy")
        if subset_choose ==1:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/valid_all.npy")
        if subset_choose ==2:
            self.data_img = np.load(self.args.root_dir+"t4sa/data_pro/test_all.npy")
        

        self.transform =  transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        
        

    def __getitem__(self,idx):
        ans_iter = int(self.label[idx])

         
            # select_class:   0:in_class, 1:out_class 
        label_list = self.class0_idx+self.class1_idx+self.class2_idx
        new_id = np.random.choice(label_list, size=1, replace=True)[0]
        
        if self.select_padding == 0:
            text = self.data_text[idx]
            text = self.sent_to_ix(text,self.token_to_ix, max_token=self.args.text_len)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)



            # print("face:",self.face_feat_list.__len__())
            
        
            # image = Image.open(self.img[idx]).convert("RGB")
            # image = self.transform(image)
            



            path = self.args.root_dir+"t4sa/"+self.data_img[new_id][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
        else:
            text = self.data_text[new_id]
            text = self.sent_to_ix(text,self.token_to_ix, max_token=self.args.text_len)
            
            text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0)

            xy_mask = torch.tensor(([False for i in range(self.args.img_len)])).unsqueeze(0)
            
            xy_mask = torch.cat((xy_mask,text_mask),dim=-1)



            # print("face:",self.face_feat_list.__len__())
            
        
            # image = Image.open(self.img[idx]).convert("RGB")
            # image = self.transform(image)
            



            path = self.args.root_dir+"t4sa/"+self.data_img[idx][1]
            img = cv2.imread(path)
            # img = img[:,:,(2,1,0)]
            img = self.transform(img)
            img =img.squeeze(0)
            
        # 49,2048
        return img,torch.from_numpy(text),text_mask,xy_mask,torch.tensor(ans_iter,dtype=torch.long)
 

    def __len__(self):
        return self.label.__len__()
       
    def feature_normalize(self,data):
        mu = np.mean(data,axis=2,keepdims=True)
        std = np.std(data,axis=2,keepdims=True)
        return (data - (mu+1e-10))/(std+1e-10)

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


class T4SA_ViT_Text(Data.Dataset):

    def __init__(self,subset_choose,token_to_ix):
        """
        subset_choose: choose train/val/test set : 0 ,1, 2,3
        """

        if subset_choose == 0:
            self.data_text = np.load("/mnt/ssd/Datasets_/t4sa/data_pro/tokens/train_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load("/mnt/ssd/Datasets_/t4sa/data_pro/labels/train.npy")
        if subset_choose ==1:
            self.data_text = np.load("/mnt/ssd/Datasets_/t4sa/data_pro/tokens/valid_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load("/mnt/ssd/Datasets_/t4sa/data_pro/labels/valid.npy")
        if subset_choose ==2:
            self.data_text = np.load("/mnt/ssd/Datasets_/t4sa/data_pro/tokens/test_token_delete_5_UNK.npy",allow_pickle=True).tolist()
            self.label = np.load("/mnt/ssd/Datasets_/t4sa/data_pro/labels/test.npy")
        


        
        
        self.text_list = self.data_text
        

        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = self.create_dict_list(self.text_list)
        
        self.vocab_size = self.token_to_ix.__len__()

        

    def __getitem__(self,idx):

        
        text = self.data_text[idx]
        text = self.sent_to_ix_cls(text,self.token_to_ix, max_token=30)
        
        text_mask = torch.tensor([True if i == 0 else False for i in text]).unsqueeze(0).unsqueeze(1)


        # print("face:",self.face_feat_list.__len__())
        ans_iter = int(self.label[idx])
       
        # image = Image.open(self.img[idx]).convert("RGB")
        # image = self.transform(image)
        
        
        # 49,2048
        return torch.from_numpy(text),text_mask,torch.tensor(ans_iter,dtype=torch.long) 
 

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
                ques_ix[ix+1] = token_to_ix[word]
            else:
                ques_ix[ix+1] = token_to_ix['UNK']

            if ix + 2 == max_token+1:
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
    data = T4SA_Vision_And_Text(0,None) 
    
    start = time.time()
    train_data_iter = DataLoader(data,batch_size=64,shuffle=True,num_workers=4) #
    for item in tqdm(train_data_iter):
        pass
    print(time.time() - start)
   