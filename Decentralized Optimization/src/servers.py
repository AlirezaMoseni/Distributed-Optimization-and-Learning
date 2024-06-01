import math
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
from tqdm import tqdm

import torch
from models import Model1, Model3
from utils import get_dataset, exp_details, setup_seed, test_inference
from clients import FedAvg_Client, FedAdmm_Client,FedProx_Client#, Scaffold_Client

class Server(object):
    def __init__(self, args):
        self.args = args
        self.history = []
        self.global_round=0
        setup_seed(args.seed)
        model = self.select_global_model(self.args.model,self.args.device)
        train_dataset, test_dataset, user_groups = get_dataset(args)
        cls = globals()[type(self).__name__.replace("_Server","_Client")]
        self.global_client = cls(args=self.args, train_set=None, test_set=test_dataset, idxs=range(len(test_dataset)), model=copy.deepcopy(model))
        self.clients = []
        for idx in range(self.args.num_users):
            self.clients.append(cls(args=self.args, train_set=train_dataset, test_set=None, idxs=user_groups[idx], model=copy.deepcopy(model)))
        if args.verbose:
            exp_details(args)
            print('random seed =', args.seed)
            print()
            print(model)

    def select_global_model(self, model, device):
        if model == 'Model1':
            global_model = Model1().to(device)
        elif model == 'Model3':
            global_model = Model3().to(device)
        else:
            exit('Error: unrecognized model')
        return global_model

    def average_weights(self,w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def run(self,frac,rounds):
        start_time = time.time()
        m = max(int(frac * self.args.num_users), 1)
        test_acc, test_loss = [], []
        train_loss = []
        for epoch in tqdm(range(rounds)):
            print(f'\n | Global Training Round : {self.global_round+1} |\n')
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            local_weights,local_losses = [],[]
            theta = self.global_client.model.state_dict()
            for i,idx in enumerate(idxs_users):
                if self.args.verbose:
                    print(" | #{:2d}: {:2d} |".format(i+1,idx))
                lsum,loss = self.clients[idx].update_weights(global_round=self.global_round, theta=copy.deepcopy(theta))
                local_weights.append(copy.deepcopy(lsum))
                local_losses.append(loss)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            mean_acc_all,_ = self.avg_trainig_calculator()
            self.update_global_model(local_weights)
            test_acc_1, test_loss_1 = self.global_client.inference("test")
            test_acc.append(test_acc_1)
            test_loss.append(test_loss_1)
            mean_train_loss = np.mean(np.array(train_loss))
            print('\ntest accuracy:{:.2f}%\n'.format(100*test_acc_1))
            print(f' \nAvg Training Stats after {self.global_round+1} global rounds:')
            print('Training Loss : {:.3f}'.format(mean_train_loss))
            self.history.append({"round":self.global_round,"test_acc":test_acc_1,"test_loss":test_loss_1,"train_loss":mean_train_loss,"train_acc":mean_acc_all})
            self.global_round+=1
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
        print(f' \n Results after {rounds} global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc[-1]))
    
    def tarining(self):
        pass
    def avg_trainig_calculator(self):
        avg_acc, avg_loss = 0.0,0.0
        self.global_client.model.eval()
        n= self.args.num_users
        for c in range(n):
            acc, loss = self.clients[c].inference("train")
            avg_acc += acc/n
            avg_loss += loss/n
        return avg_acc, avg_loss
    
    def plot(self):
        n=self.args.num_users
        s = math.ceil(math.sqrt(n))
        fig, axs = plt.subplots(2*s, s,figsize=(s*2,s*4),sharex=True,sharey="row")
        axs = axs.flat
        for i in range(0,2*n,2*s):
            for j in range(s):
                hist = pd.DataFrame(self.clients[i//2+j].history)
                axs[i+j].set_title("Client #%d" % (i//2+j+1))
                axs[i+j].set(xlabel='rounds', ylabel='loss')
                axs[i+j+10].set(xlabel='rounds', ylabel='accuracy')
                axs[i+j].label_outer()
                axs[i+j+10].label_outer()
                try:
                    axs[i+j].plot(hist["train_loss"],"b",label="train")
                    axs[i+j].plot(hist["val_loss"],"r",label="val")
                    axs[i+j].legend()
                except:
                    pass
                try:
                    axs[i+j+10].plot(hist["train_acc"],"k",label="train")
                    axs[i+j+10].plot(hist["val_acc"],"g",label="val")
                    axs[i+j+10].legend()
                except:
                    pass
        plt.show()
class FedAdmm_Server(Server):
    def __init__(self, args):
        super().__init__(args)

    def update_global_model(self,local_sum):
        theta = self.average_weights(local_sum)
        self.global_client.model.load_state_dict(theta)

class FedAvg_Server(Server):
    def __init__(self,args):
        super().__init__(args)
    
    def update_global_model(self,local_weights):
        global_weights = self.average_weights(local_weights)
        self.global_client.model.load_state_dict(global_weights)

class FedProx_Server(Server):
    def __init__(self,args):
        super().__init__(args)
    
    def update_global_model(self,local_weights):
        global_weights = self.average_weights(local_weights)
        self.global_client.model.load_state_dict(global_weights)