from gcn import create_gcn_model
from train import run
from data import load_data

if __name__=='__main__':
    data=load_data('cora')
    model,optimizer=create_gcn_model(data)
