import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
from utils import DatasetSplit

class Client(object):
    def __init__(self, args, train_set, test_set, idxs, model):
        self.args = args
        self.loaders = self.train_val_test(train_set, test_set, idxs)
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.history = []
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train_val_test(self,  train_set, test_set, idxs):
        if test_set:
            idxs_test = range(len(test_set))
            test_data = DatasetSplit(test_set, idxs_test)
            return {
                "train": None,
                "test": DataLoader(test_data, batch_size=self.args.local_bs, shuffle=False),
            }
        else:
            idxs = list(idxs)
            val_size = max(int(len(idxs)/10), 1)
            idxs_train = idxs[val_size:]
            idxs_test = idxs[:val_size]
            train_data = DatasetSplit(train_set, idxs_train)
            test_data = DatasetSplit(train_set, idxs_test)
            return {
                "train": DataLoader(train_data, batch_size=self.args.local_bs, shuffle=True),
                "test":  DataLoader(test_data, batch_size=self.args.local_bs, shuffle=False),
            }

    def update_weights(self, theta, global_round):
        self.model.load_state_dict(theta)
        epoch_loss = 0.0
        for it in range(self.args.local_ep):
            train_acc,train_loss = 0.0,[]
            total = len(self.loaders["train"].dataset)
            for images, labels in self.loaders["train"]:
                loss,corr = self.update_model(images, labels,theta)
                self.optimizer.step()
                train_loss.append(loss.item())
                train_acc += corr/total
            val_acc, val_loss = self.inference("test")
            train_loss = sum(train_loss)/len(train_loss)
            self.report(global_round, it, train_loss,train_acc,val_acc, val_loss)
            self.history.append({"global_round": global_round, "epoch": it,"train_loss": train_loss,"train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss})
            epoch_loss += train_loss/self.args.local_ep
        self.update_duals(theta)
        return self.model.state_dict(), epoch_loss
    
    def update_model(self):
        pass
    
    def update_duals(self,theta):
        pass
    
    def inference(self,dataset):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
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
        accuracy = correct/total
        return accuracy, loss

    def report(self, global_round, it, train_loss,train_acc,val_acc, val_loss):
        if self.args.verbose and (it % 9 == 0):
            print('| Local Epoch : {:2d} | Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}% |'.format(it+1, train_loss,train_acc*100,val_loss,val_acc*100))

class FedAvg_Client(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def update_model(self,images, labels,theta):
        images, labels = images.to(self.args.device), labels.to(self.args.device)
        self.model.zero_grad()
        self.optimizer.zero_grad()
        log_probs = self.model(images)
        loss = self.criterion(log_probs, labels)
        loss.backward()
        _, pred_labels = torch.max(log_probs, 1)
        pred_labels = pred_labels.view(-1)
        correct = torch.sum(torch.eq(pred_labels, labels)).item()
        return loss,correct

class FedProx_Client(Client):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def update_model(self,images, labels,theta):
        images, labels = images.to(self.args.device), labels.to(self.args.device)
        self.model.zero_grad()
        self.optimizer.zero_grad()
        log_probs = self.model(images)
        loss = self.criterion(log_probs, labels)
        loss.backward()
        model_weights_pre = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                param.grad = param.grad + self.args.rho * (model_weights_pre[name]-theta[name]) #proximal regularization
        _, pred_labels = torch.max(log_probs, 1)
        pred_labels = pred_labels.view(-1)
        correct = torch.sum(torch.eq(pred_labels, labels)).item()
        return loss,correct

class FedAdmm_Client(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        theta = self.model.state_dict()
        self.alpha = {}
        for key in theta.keys():
            self.alpha[key] = torch.zeros_like(theta[key]).to(self.args.device)
        
    def update_model(self,images,labels,theta):
        images, labels = images.to(self.args.device), labels.to(self.args.device)
        self.optimizer.zero_grad()
        self.model.zero_grad()
        log_probs = self.model(images)
        loss = self.criterion(log_probs, labels)
        loss.backward()
        model_weights_pre = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                param.grad = param.grad + (self.alpha[name] + self.args.rho * (model_weights_pre[name]-theta[name]))
        _, pred_labels = torch.max(log_probs, 1)
        pred_labels = pred_labels.view(-1)
        correct = torch.sum(torch.eq(pred_labels, labels)).item()
        return loss,correct

    def update_duals(self,theta):
        weights = self.model.state_dict()
        for key in self.alpha.keys():
            self.alpha[key] = self.alpha[key] + self.args.rho * (weights[key]-theta[key])

# class Scaffold_Client(Client):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)

#     def update_weights(self, model, global_round, alpha, alpha_server, theta):


#                 for name, param in model.named_parameters():
#                     if param.requires_grad == True:
#                         param.grad = param.grad + alpha_server[name] - alpha[name]
        
#         local_sum = {}
#         control_update = {}
#         weights = model.state_dict()
#         alpha_temp = copy.deepcopy(alpha)
#         for key in alpha.keys():
#             local_sum[key] = weights[key] - theta[key]
#         for key in alpha_temp.keys():
#             a = 1/ (self.args.local_num * self.args.lr) 
#             alpha_temp[key] = alpha[key] - alpha_server[key] - local_sum[key] * a
#         for key in alpha.keys():
#             control_update[key] = alpha_temp[key] - alpha[key]
#             alpha[key] = alpha_temp[key]
        
#         return sum(epoch_loss) / len(epoch_loss), local_sum, control_update, model, alpha