
import torch
import torch.utils.data.dataset as Data  
from torch.utils.data import DataLoader,Subset,Dataset
import numpy as np 
import pandas as pd 
import re
import en_vectors_web_lg
import torchvision.transforms as transforms
# from nltk import PorterStemmer


def get_transforms(args):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def sent_to_ix(s, token_to_ix, max_token=100):
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


def sent_to_ix_concat(s1,s2, token_to_ix, max_token=100):
    ques_ix = np.zeros(max_token*2+1, np.int64)

    for ix, word in enumerate(s1):
        #word = p.singular_noun(word)
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break
    ques_ix[max_token] = token_to_ix['SEP']

    for ix, word in enumerate(s2):
        #word = p.singular_noun(word)
        if word in token_to_ix:
            ques_ix[max_token+ix+1] = token_to_ix[word]
        else:
            ques_ix[max_token+ix+1] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break


    return ques_ix

def tokenize_list(key_to_word):
    sentence_list = []
    for v in key_to_word:
        sentence_list.append([clean(w) for w in v.split() if clean(w) != ''])
    return sentence_list

def create_dict_list(key_to_sentence, use_glove=True):
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
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('SEP').vector)


    for v in key_to_sentence:
        for word in v:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    # np.save(glove_file, pretrained_emb)
    # pickle.dump(token_to_ix, open(token_file, "wb"))
    return token_to_ix, pretrained_emb


def clean(w):
    return re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            w.lower()
            ).replace('-', ' ').replace('/', ' ')


def str_to_label(name):
    """
        0 : neutral
        1 : contradiction
        2 : entailment
    """
    if name == "neutral":
        return 0
    if name== "contradiction":
        return 1
    if name== "entailment":
        return 2 


## twitter text filter
def filter_emoji(content):

    try:
        cont = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        cont = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return cont.sub(u'', content)

def clean(w):
    w = re.sub(
            r"([,'!?\"()*#:;\[\]])",
            '',
            w.lower()
            ).replace('-', ' ').replace('/', ' ')

    w = re.sub('[^\w\u4e00-\u9fff]+', '',w)
    w = re.sub(r'[^a-zA-Z]',' ',w) 
    return w 



def filter_data(sentence,if_stem,delete_list):
    item = sentence.split()

    item_new = [ii for ii in item if "http" not in ii]
    
    item_new = [ii for ii in item_new if "@" not in ii]

    item_new = [ii for ii in item_new if "#" not in ii]
    
    item_new = " ".join(item_new)
    item_new = re.sub(
                r"([.!?\"()*#:;])",
                ' ',
                item_new
            ).replace('-',' ').replace('—',' ').replace('/',' ').replace('【',' ').replace('】',' ')
    item_new =  re.sub(r'[0-9]+', ' ', item_new)
    
    w = [filter_emoji(i) for i in item_new.split()]

    if if_stem==True:
        porter = PorterStemmer()
        w = [porter.stem(t) for t in w]
    
    w = [clean(t) for t in w]
    w = [i for i in w if i!=""]

    ## 删除单字母单词
    temp_w = []
    for i in w:
        if i.__len__()!=1:
            temp_w.append(i)
        else:
            if i in ['a','i','u']:
                temp_w.append(i)
            else:
                pass 
    
    ## 删除出现次数小于5的单词
    #w = [i for i in temp_w if i not in delete_list]

    return w 