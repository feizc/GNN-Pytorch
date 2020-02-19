import torch
import  torch.nn.functional as F
from numpy import mean, std
from tqdm import tqdm

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping(object):
    def __init__(self,patience,verbose):
        self.patience=patience
        self.verbose=verbose
        self.counter=0
        self.best_val_loss=float('inf')

    def check(self,evals,epoch):
        if evals['eval_loss']<self.best_val_loss:
            self.best_val_loss=evals['eval_loss']
            self.counter=0
        else:
            self.counter+=1
        stop=False
        if self.counter>=self.patience:
            stop=True
            if self.verbose:
                print("Stop training, epoch: ",epoch)
        return stop

def train(model,optimizer,data):
    model.train()
    optimizer.zero_grad()
    output=model(data)
    loss=F.nll_loss(output[data.train_mask],data.labels[data.train_mask])
    loss.backward()
    optimizer.step()

def evaluate(model,data):
    model.eval()
    with torch.no_grad():
        output=model(data)

    outputs={}
    for key in ['train','val','test']:
        if key=='train':
            mask=data.train_mask
        if key=='val':
            mask=data.val_mask
        else:
            mask=data.test_mask
        loss=F.nll_loss(output[mask],data.labels[mask]).item()
        pred=output[mask].max(dim=1)[1]
        acc=pred.eq(data.labels[mask]).sum().item()/mask.sum().item()

        outputs['{}_loss'.format(key)]=loss
        output['{}_acc'.format(key)]=acc
    return outputs


def run(data, model, optimizer, epochs=200, niter=100, early_stopping=True, patience=10, verbose=False):
    # for GPU
    data.to(device)

    val_acc_list = []
    test_acc_list = []

    for _ in tqdm(range(niter)):
        model.to(device).reset_parameters()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # for early stopping
        if early_stopping:
            stop_checker = EarlyStopping(patience, verbose)

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            evals = evaluate(model, data)

            if verbose:
                print('epoch: {: 4d}'.format(epoch),
                      'train loss: {:.5f}'.format(evals['train_loss']),
                      'train acc: {:.5f}'.format(evals['train_acc']),
                      'val loss: {:.5f}'.format(evals['val_loss']),
                      'val acc: {:.5f}'.format(evals['val_acc']))

            if early_stopping:
                if stop_checker.check(evals, epoch):
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        evals = evaluate(model, data)
        if verbose:
            for met, val in evals.items():
                print(met, val)

        val_acc_list.append(evals['val_acc'])
        test_acc_list.append(evals['test_acc'])

    print("mean", mean(test_acc_list))
    print("std", std(test_acc_list))
    return {
        'val_acc': mean(val_acc_list),
        'test_acc': mean(test_acc_list),
        'test_acc_std': std(test_acc_list)
    }