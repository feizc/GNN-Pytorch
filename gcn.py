import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.modules import Module

class GCN(nn.Module):
    def __init__(self,data,hid_dim,dropout):
        super(GCN,self).__init__()
        feature_dim,class_dim=data.feature_dim, data.class_dim
        self.gc1=GCNConv(feature_dim,hid_dim)
        self.gc2=GCNConv(hid_dim,class_dim)
        self.dropout=dropout

    def reset_parameter(self):
        self.gc1.reset_parameter()
        self.gc2.reset_parameter()

    def forward(self, data):
        x=data.features
        adj=data.adj
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=F.relu(self.gc1(x,adj))
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gc2(x,adj)
        return F.log_softmax(x,dim=1)

class GCNConv(Module):
    def __init__(self,input_dim,output_dim,bias=True):
        super(GCNConv,self).__init()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight=nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        if bias:
            self.bias=nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.weight.data,gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self,x,adj):
        x=torch.matmul(x, self.weight)
        x=torch.spmm(adj,x)
        if self.bias is not None:
            x+=self.bias
        return x

def create_gcn_model(data, hid_dim=16, dropout=0.5,learning_rate=0.01,weight_decay=5e-4):
    model=GCN(data,hid_dim,dropout)
    optimizer=Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    return model,optimizer