import torch
import math
import json
import torchvision
from torch import nn
from torchvision import models


class AddNorm(nn.Module):
    def __init__(self,num_hiddens,dropout,**kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(num_hiddens)

    def forward(self,x,y):
        return self.ln(self.dropout(y)+x)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout)
        self.Wq=nn.Linear(query_size,num_hiddens,bias=bias)
        self.Wk=nn.Linear(key_size,num_hiddens,bias=bias)
        self.Wv=nn.Linear(value_size,num_hiddens,bias=bias)
        self.Wo=nn.Linear(num_hiddens,num_hiddens,bias=bias)


    def transpose_qkv(self,x,num_heads):
        x=x.reshape(x.shape[0],x.shape[1],num_heads,-1)
        x=x.permute(0,2,1,3)
        return x.reshape(-1,x.shape[2],x.shape[3])

    def transpose_output(self,x,num_heads):
        x=x.reshape(-1,num_heads,x.shape[1],x.shape[2])
        x=x.permute(0,2,1,3)
        return x.reshape(x.shape[0],x.shape[1],-1)
    
    def forward(self,queries,keys,values):
        queries=self.transpose_qkv(self.Wq(queries),self.num_heads)
        keys=self.transpose_qkv(self.Wk(keys),self.num_heads)
        values=self.transpose_qkv(self.Wv(values),self.num_heads)

        output=self.attention(queries,keys,values)
        output=self.transpose_output(output,self.num_heads) 

        return self.Wo(output)

class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,ffn_num_hiddens,dropout,**kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention=MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout)
        self.addnorm1=AddNorm(num_hiddens,dropout)
        self.ffn=nn.Sequential(
            nn.Linear(num_hiddens,ffn_num_hiddens),
            nn.ReLU(),
            nn.Linear(ffn_num_hiddens,num_hiddens)
            )
        self.addnorm2=AddNorm(num_hiddens,dropout)       

    def forward(self,x):   
        y=self.attention(x,x,x)
        x=self.addnorm1(x,y)
        y=self.ffn(x)
        return self.addnorm2(x,y) 
    
class FlowClassifier(nn.Module):
    def __init__(self,config_file,**kwargs):
        super(FlowClassifier, self).__init__(**kwargs)
        with open(config_file, 'r') as f:
            config= json.load(f)["model"]
            
        self.embedding = nn.Linear(1,config["encoder"]["num_hiddens"],dtype=torch.float32)
        self.blks=nn.Sequential()
        for i in range(config["encoders_layers"]):
            self.blks.add_module("encoderblock"+str(i),
                                 EncoderBlock(**config["encoder"]))
        
        self.classifier = nn.Sequential(
            nn.Dropout(config["classifier"]["dropout1"]),
            nn.Linear(config["encoder"]["num_hiddens"],config["classifier"]["classes_num_hiddens"]),
            nn.ReLU(inplace=True),
            nn.Dropout(config["classifier"]["dropout2"]),
            nn.Linear(config["classifier"]["classes_num_hiddens"],config["classifier"]["num_classes"])
        )

    def forward(self, x):
        x=x.unsqueeze(-1)
        x=self.embedding(x) 
        for blk in self.blks:
            x=blk(x)
        x=torch.mean(x,dim=1)
        return self.classifier(x)











