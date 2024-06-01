import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import DatasetSplit
import copy
import numpy as np

class Client():

    def __init__(self, args, train_set, test_set, idxs, model):
        self.args = args
        self.loaders = self.train_val_test(train_set, test_set, idxs)
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.history = {}
        self.rounds = 1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train_val_test(self,  train_set, test_set, idxs):
        val_size = max(int(len(idxs)/10), 1)
        rand_set = set(np.random.choice(list(idxs),val_size, replace=False))
        idx_train = list(set(idxs) - rand_set)

        # idxs_train = idxs[val_size:]
        # idxs_val = idxs[:val_size]
        train_data = DatasetSplit(train_set, idx_train)
        val_data   = DatasetSplit(train_set, rand_set)
        return {
            "train": DataLoader(train_data, batch_size=self.args.local_bs, shuffle=True ),
            "val":   DataLoader(val_data,   batch_size=self.args.local_bs, shuffle=True),
            "test":  DataLoader(test_set,   batch_size=self.args.local_bs, shuffle=True),
        }

    def local_update(self, global_round):
        epoch_loss = 0.0
        train_hist = []
        for it in range(self.args.local_ep):
            train_acc, train_loss = 0.0, []
            total = len(self.loaders["train"].dataset)
            for images, labels in self.loaders["train"]:
                self.model.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                ___, pred_labels = torch.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)
                correct = torch.sum(torch.eq(pred_labels, labels)).item()
                self.optimizer.step()
                train_loss.append(loss.item())
                train_acc += correct/total
            val_acc, val_loss = self.inference("val")
            train_loss = sum(train_loss)/len(train_loss)
            self.report(it, train_loss, train_acc, val_acc, val_loss)
            train_hist.append({"iter": it,"train_loss": train_loss,"train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss})
            epoch_loss += train_loss/self.args.local_ep
        self.history[global_round]["train_hist"]=train_hist
        self.rounds += 1
        return epoch_loss

    def consensus(self,Ni):
        w_avg = {}
        weights = self.model.state_dict()
        for key in weights.keys():
            w_avg[key]=torch.zeros_like(weights[key])
        for aij, weights in Ni:
            for key in weights.keys():
                w_avg[key] += torch.mul(weights[key],aij)    
        return w_avg

    def inference(self,dataset):
        self.model.eval()
        bs ,loss, total, correct = 0.0, 0.0, 0.0,0.0
        for images, labels in self.loaders[dataset]:
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            bs+=1
        accuracy = correct/total
        return accuracy, loss/bs

    def report(self, it, train_loss, train_acc, val_acc, val_loss):
        if self.args.verbose:# and (it % 9 == 0):
            print('| Local Epoch : {:2d} | Train Loss: {:2.3f} | Train Acc: {:4.2f}% | Val Loss: {:2.3f} | Val Acc: {:4.2f}% |'.format(it+1, train_loss, train_acc*100, val_loss, val_acc*100))
            # print('| Local Epoch : {:2d} | Train Loss: {:3f} | Train Acc: {:3f}% | Val Loss: {:3f} | Val Acc: {:3f}% |'.format(it+1, train_loss, train_acc*100, val_loss, val_acc*100))        