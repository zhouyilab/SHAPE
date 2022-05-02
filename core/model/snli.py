import math
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.modules.linear import Linear
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from core.model.attention import *



# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    """ 
        in: b,seq_len,dim 
        out: b,1,1,seq_len 
             
        padding: True
        value: False 
    
    """
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

def make_mask_cls(feature):
    """ 
        in: b,seq_len,dim 
        out: b,1,1,seq_len 
             
        padding: True
        value: False 
    
    """
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

# ------------------------------
# ---------- Flattening --------
# ------------------------------


# ------------------------
# ---- Self Attention ----
# ------------------------
# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            #scores = self.get_attn_pad_mask(query, key)
            scores = scores.masked_fill(mask, -1e9)
    
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)



class SoftGate(nn.Module):
    def __init__(self,args):
        super(SoftGate, self).__init__()
        """
        padding version: img_len = text_len 

        """
        self.args = args
        self.conv1d = nn.Conv1d(self.args.dim,10,1,1)

        self.max_pool = nn.MaxPool2d((5,1))

        self.layer = nn.Sequential(
            nn.Linear(2*(self.args.text_len+self.args.text_len),64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,2)
        )
        self.sa = MHAtt(self.args)


    def forward(self,x,y,xy_mask):
        xy = torch.cat((x,y),1) # b,len_x+len_y,dim 
        out = self.conv1d(xy.permute(0,2,1)) # b,10,len_x+len_Y
        out = self.max_pool(out) # b,2,len_x+len_y
       
        out = torch.flatten(out,1) # b,2*(len_x+len_y)
        out = self.layer(out) #b,1 
        lamb = torch.sigmoid(out)
        
        lamb_x = lamb[:,0].unsqueeze(1).unsqueeze(2)
        lamb_y = lamb[:,1].unsqueeze(1).unsqueeze(2)

      
        co_representation = self.sa(xy,xy,xy,xy_mask) #b,len_x+len_y,dim
    
        x_representation = co_representation[:,0:self.args.text_len,:]
        y_representation = co_representation[:,self.args.text_len:,:]

        
        return lamb_x,lamb_y,x_representation,y_representation


class HardGate(nn.Module):
    def __init__(self,args):
        pass 
    def forward(self,x,y):
        pass 



class FeedLayer(nn.Module):
    def __init__(self, args):
        super(FeedLayer, self).__init__()

        #self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        # self.dropout2 = nn.Dropout(args.dropout_r)
        # self.norm2 = LayerNorm(args.hidden_size)

        

    def forward(self, y, y_new):
        y = self.norm1(y_new + self.dropout1(
            self.ffn(y)
        ))

        return y





# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

# ---------------------------
# ---- FF + norm  -----------
# ---------------------------
class FFAndNorm(nn.Module):
    def __init__(self, args):
        super(FFAndNorm, self).__init__()

        self.ffn = FFN(args)
        self.norm1 = LayerNorm(args.hidden_size)
        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, x):
        x = self.norm1(x)
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class FilterGate(nn.Module):
    def __init__(self,args):
        super(FilterGate,self).__init__()
        self.args = args 
        self.linear = nn.Linear(args.hidden_size*2,1)
    def forward(self,x_input,y_input,x_output):
        """
        x_input: b,seq_len, hidden_size
        y_input: b,seq_len, hidden_size
        x_output: b,seq_len, hidden_size
        """

        # b, seq_len, hidden_size*2 
        feature_concat = torch.concat((x_input,y_input),dim=2)
        out = self.linear(x_input) # b,seq_len,1 
        out = F.sigmoid(out)
        

        return out,1-out 

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()

        self.mhatt = MHAtt(args)
        #self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        # self.dropout2 = nn.Dropout(args.dropout_r)
        # self.norm2 = LayerNorm(args.hidden_size)

        

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        # y = self.norm2(y + self.dropout2(
        #     self.ffn(y)
        # ))

        return y

class Block(nn.Module):
    def __init__(self, args, i,m_select):
        super(Block, self).__init__()
        self.args = args
        self.m_select = m_select
        self.mhatt_x = MHAtt(args)
        #self.ffn = FFN(args)

        self.dropout_x = nn.Dropout(args.dropout_r)
        self.norm_x = LayerNorm(args.hidden_size)

        self.mhatt_y = MHAtt(args)
        #self.ffn = FFN(args)

        self.dropout_y = nn.Dropout(args.dropout_r)
        self.norm_y = LayerNorm(args.hidden_size)





        self.fb1 = FeedLayer(args)
        self.fb2 = FeedLayer(args)

        self.softgate = SoftGate(args)

    def forward(self, x, x_mask, y, y_mask,xy_mask):
        """
            dynamic fusion in attention component
        """
        

        lamb_x,lamb_y,x_representation,y_representation = self.softgate(x,y,xy_mask) 

        if self.m_select =="Dynamic":
            ax = self.norm_x(x_representation*lamb_x+x*(1-lamb_x) + self.dropout_x(
                self.mhatt_x(x, x, x, x_mask)
            ))

            ay = self.norm_y(y_representation*lamb_y+y*(1-lamb_y) + self.dropout_y(
                self.mhatt_y(y, y, y, y_mask)
            ))
        elif self.m_select=="Dense":
            ax = self.norm_x(x_representation + self.dropout_x(
                self.mhatt_x(x, x, x, x_mask)
            ))

            ay = self.norm_y(y_representation + self.dropout_y(
                self.mhatt_y(y, y, y, y_mask)
            ))
        else:  # Late
            ax = self.norm_x(x + self.dropout_x(
            self.mhatt_x(x, x, x, x_mask)
            ))

            ay = self.norm_y(y + self.dropout_y(
                self.mhatt_y(y, y, y, y_mask)
            ))

        ax = self.fb1(ax,ax)
        ay = self.fb2(ay,ay)

      

        return ax,ay 







class SNLI(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        """
        text padding version; 
        """

        super().__init__()
        self.agrs = args 

        self.m_select = args.model

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )
        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))



        self.adapter_text_pre = nn.Linear(args.word_embed_size,args.dim)
        self.adapter_text_hyp = nn.Linear(args.word_embed_size,args.dim)

        
        self.pos_embedding_text_pre = nn.Parameter(torch.randn(1, args.text_len, args.dim))
        

        self.pos_embedding_text_hyp = nn.Parameter(torch.randn(1, args.text_len, args.dim))
        

        
        
        
        self.dropout = nn.Dropout(0.1)


        self.to_latent = nn.Identity()

        self.mlp_concat = nn.Sequential(
            nn.LayerNorm(args.dim),
            nn.Linear(args.dim, args.num_classes)
        )


        self.enc_list = nn.ModuleList([Block(args, i,self.m_select) for i in range(args.layer)])

    def forward(self,pre,hyp,pre_mask,hyp_mask,xy_mask):
       
        # test : b,
        # x_mask =pre_mask.unsqueeze(1)
        # y_mask =hyp_mask.unsqueeze(1)
        # xy_mask = xy_mask.unsqueeze(1)


        pre = self.embedding(pre)
        pre = self.adapter_text_pre(pre)
        # 1,1,d --> b,1,d 
        
        # 1, num_of_patch+1,dim  # broadcast for each batch 
        x =pre+ self.pos_embedding_text_pre[:, :]
        x = self.dropout(x)  # batch,65,1024



        # hyp embedding 
        hyp = self.embedding(hyp)
        hyp = self.adapter_text_pre(hyp)
        # 1,1,d --> b,1,d 
       
        # 1, num_of_patch+1,dim  # broadcast for each batch 
        y =hyp+ self.pos_embedding_text_hyp[:, :]
        y = self.dropout(y)  # batch,65,1024


        

        # for i, dec in enumerate(self.enc_list):
        #         x_m, y_m = None, None
        #         if i == 0:
        #             x_m, y_m = None, y_mask
        #         x, y = dec(x, x_m, y, y_m,xy_mask)
        # print(x.shape,y.shape,x_mask.shape,y_mask.shape)
        for i, dec in enumerate(self.enc_list):
                # x_m, y_m = None, None
                # if i == 0:
                x_m, y_m = pre_mask,hyp_mask
                x, y = dec(x, x_m, y, y_m, xy_mask)



        x = x[:, 0]
        x = self.to_latent(x) 

        
        y = y[:,0]
        y = self.to_latent(y)

        out = x+y 

        return self.mlp_concat(out)

    def feedforward(self,device,feat,net):
        x,y,x_mask,y_mask,xy_mask,ans = feat
        x = x.to(device)
        y = y.to(device)
        x_mask = x_mask.to(device)
        y_mask = y_mask.to(device)
        xy_mask = xy_mask.to(device)
        ans = ans.to(device)

        out = net(x,y,x_mask,y_mask,xy_mask)
        return out,ans 

    def feedforward_padding(self,device,feat,net,select_padding):
        x,y,x_mask,y_mask,xy_mask,ans = feat
        if select_padding ==0:
            x = torch.zeros_like(x).to(device)   
        else:
            y = torch.zeros_like(y).to(device)

            
        x = x.to(device)
        y = y.to(device)
        x_mask = x_mask.to(device)
        y_mask = y_mask.to(device)
        xy_mask = xy_mask.to(device)
        ans = ans.to(device)

        out = net(x,y,x_mask,y_mask,xy_mask)
        return out,ans 