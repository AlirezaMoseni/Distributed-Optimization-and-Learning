import copy
import torch
import time
from models import Model1, Model3
from utils import setup_seed, exp_details, get_dataset
from clients import Client
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class Simulator(object):
    def __init__(self, args):
        self.args = args
        self.history = []
        self.global_round = 0
        setup_seed(args.seed)
        model = self.select_global_model(self.args.model, self.args.device)
        train_dataset, test_dataset, user_groups = get_dataset(args)
        self.clients = []
        self.adjacent_matrix = self.communication_graph(args.topology,args.mode,args.num_users)
        for idx in range(self.args.num_users):
            self.clients.append(Client(args=self.args, train_set=train_dataset,test_set=test_dataset, idxs=user_groups[idx], model=copy.deepcopy(model)))
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

    def communication_graph(self,topology,mode,n):
        graphs = []
        if   topology == "circle":
            g = np.zeros((n,n))
            for i in range(n):
                g[i][(i+1)%n]=1
                g[(i+1)%n][i]=1
            graphs.append(g)
        elif topology == "star":
            g = np.zeros((n,n))
            for i in range(n-1):
                g[i+1][0]=1
                g[0][i+1]=1
            graphs.append(g)
        elif topology == "compelete":
            g = np.ones((n,n))
            for i in range(n):
                g[i][i]=0
            graphs.append(g)
        elif topology == "dynamic":
            for i in range(n):
                g_t = np.zeros((n, n))
                g_t[i, (i+1)%n] = 1
                g_t[(i+1)%n, i] = 1
                graphs.append(g_t)        
        if mode == "stochastic":
            rand_graph = torch.rand(n,n)
            for i in range(len(graphs)):
                graph = rand_graph* torch.tensor(graphs[i]).int().float()
                graph /= graph.sum(0)
                graphs[i] = graph.T
        elif mode == "double_stochastic":
            rand_graph = torch.rand(n,n)
            if topology == "star":
                rand_graph = torch.ones(n,n)/n
            for i in range(len(graphs)):
                graph = np.array(rand_graph* torch.tensor(graphs[i]).int().float())
                rsum = graph.sum(1)
                csum = graph.sum(0)
                print(rsum,csum)
                while (np.any(rsum != 1)) | (np.any(csum != 1)):
                    graph /= graph.sum(0)
                    graph = graph / graph.sum(1)[:, np.newaxis]
                    rsum = graph.sum(1)
                    csum = graph.sum(0)
                graphs[i] = torch.tensor(graph).T
        return graphs

    def run(self, rounds):
        pass

    def Neighbors(self,i,graph) :
        Ni = []
        for j in range(self.args.num_users):
            aij = graph[i][j]
            if aij>0:
                Ni.append((aij,self.clients[j].model.state_dict()))
        return Ni

    def report(self,local_losses,test_loss_1,test_acc_1):
        loss_avg = sum(local_losses) / len(local_losses)
        test_loss_avg = sum(test_loss_1) / len(test_loss_1)
        test_acc_avg = sum(test_acc_1) / len(test_acc_1)
        print(f' \nAvg Training Stats after {self.global_round+1} global rounds:')
        print('Training Loss : {:.3f}'.format(loss_avg))
        print('Test Loss : {:.3f}'.format(test_loss_avg))
        print('Test ACC : {:.2f}'.format(test_acc_avg))
        self.history.append({"round": self.global_round, "avg_test_acc": test_acc_avg,"avg_test_loss": test_loss_avg, "avg_train_loss": loss_avg})
        self.global_round += 1
        
class NoConsDecFedAvg(Simulator):
    def __init__(self, args):
        super().__init__(args)

    def run(self, rounds):
        start_time = time.time()
        for rounds in tqdm(range(rounds)):
            print(f'\n | Local Training Round : {self.global_round+1} |\n')
            local_losses,test_acc_1,test_loss_1 = [],[],[]
            for i,client in enumerate(self.clients):
                if self.args.verbose:
                    print(f" | #{i+1:2d} |")
                client.history[self.global_round] ={}
                loss = client.local_update(global_round=self.global_round)
                local_losses.append(loss)
                test_acc, test_loss = client.inference("test")
                test_acc_1.append(test_acc)
                test_loss_1.append(test_loss)
                client.history[self.global_round]["test_hist"] = {"test_loss":test_loss,"test_acc":test_acc}
                print(f"| After  Local Update | Test Loss : {test_loss:2.3f} | Test ACC: {test_acc:4.3f} |")
            self.report(local_losses,test_loss_1,test_acc_1)
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

class DecFedAvg(Simulator):
    def __init__(self, args):
        super().__init__(args)
    def run(self, rounds):
        start_time = time.time()
        for rounds in tqdm(range(rounds)):
            print(f'\n | Local Training Round : {self.global_round+1} |\n')
            local_losses,test_acc_1,test_loss_1 = [],[],[]
            t = self.global_round%len(self.adjacent_matrix)
            graph = self.adjacent_matrix[t]
            print(f'\n | Communication Graph')
            for row in graph:
                print(row.numpy())
            print()
            new_weights=[]
            for i,client in enumerate(self.clients):
                Ni = self.Neighbors(i,graph)
                new_weights.append(client.consensus(Ni=Ni))
            for new_w, client in zip(new_weights,self.clients):
                client.model.load_state_dict(new_w)
                test_acc, test_loss = client.inference("test")
                client.history[self.global_round] ={}
                client.history[self.global_round]["test_hist"] = {"test_loss":test_loss,"test_acc":test_acc}
                test_acc_1.append(test_acc)
                test_loss_1.append(test_loss)
                print(f"| Before Local Update | Test Loss : {test_loss:2.3f} | Test ACC: {test_acc:4.3f} |")
            for i,client in enumerate(self.clients):
                if self.args.verbose:
                    print(f" | #{i+1:2d} |")
                loss = client.local_update(global_round=self.global_round)
                local_losses.append(loss)
                test_acc, test_loss = client.inference("test")
                print(f"| After  Local Update | Test Loss : {test_loss:2.3f} | Test ACC: {test_acc:4.3f} |")
            self.report(local_losses,test_loss_1,test_acc_1)
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

class Centeralized(NoConsDecFedAvg):
    def __init__(self, args):
        args.num_users = 1
        args.local_ep = 1
        args.iid = True
        super().__init__(args)

class FedLCon(Simulator):
    def __init__(self, args):
        args.num_users = 1
        args.local_ep = 1
        args.iid = True
        super().__init__(args)
    def run(self, rounds,eps):
        start_time = time.time()
        for rounds in tqdm(range(rounds)):
            print(f'\n | Local Training Round : {self.global_round+1} |\n')
            local_losses,test_acc_1,test_loss_1 = [],[],[]
            t = self.global_round%len(self.adjacent_matrix)
            graph = self.adjacent_matrix[t]
            new_weights=[]
            for j in range(eps):
                print(f"\n| Consesnsus Round {j} |")
                for i,client in enumerate(self.clients):
                    Ni = self.Neighbors(i,graph)
                    new_weights.append(client.consensus(Ni=Ni))
                for new_w, client in zip(new_weights,self.clients):
                    client.model.load_state_dict(new_w)
            for client in self.clients:
                test_acc, test_loss = client.inference("test")
                client.history[self.global_round] ={}
                client.history[self.global_round]["test_hist"] = {"test_loss":test_loss,"test_acc":test_acc}
                test_acc_1.append(test_acc)
                test_loss_1.append(test_loss)
                print(f" Test Loss : {test_loss:2.3f} | Test ACC: {test_acc:4.3f} |")
            for i,client in enumerate(self.clients):
                if self.args.verbose:
                    print(f" | #{i+1:2d} |")
                loss = client.local_update(global_round=self.global_round)
                local_losses.append(loss)
                test_acc, test_loss = client.inference("test")
                print(f"| After  Local Update | Test Loss : {test_loss:2.3f} | Test ACC: {test_acc:4.3f} |")
            self.report(local_losses,test_loss_1,test_acc_1)
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


class GossipLearning(Simulator):
    def __init__(self, args):
        super().__init__(args)
