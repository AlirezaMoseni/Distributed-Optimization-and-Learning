import copy
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid
from sampling import cifar_iid, cifar_noniid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

def servers_plot(servers,clients,frac,iid):
    title = "| {} Clients | frac: {} | iid: {} |".format(clients,frac,iid)
    fig, axs = plt.subplots(2, 2, figsize=(30, 15))
    fig.suptitle(title, fontsize=36)
    axs[0, 0].set_title('Average train accuracy of all clients', fontsize=22)
    axs[0, 0].set_ylabel('Average Accuracy')
    axs[0, 1].set_title('Average training loss of clients in a round', fontsize=22)
    axs[0, 1].set_ylabel('Training loss')
    axs[1, 0].set_title('Test Accuracy of Global model', fontsize=22)
    axs[1, 0].set_ylabel('test_acc')
    axs[1, 1].set_title('Test Loss of Global model', fontsize=22)
    axs[1, 1].set_ylabel('test_loss')
    for server in servers:
        hist = pd.DataFrame(server.history)
        name = type(server).__name__.replace("_Server","")
        axs[0, 0].plot(hist["train_acc"],label=name)
        axs[0, 1].plot(hist["train_loss"],label=name)
        axs[1, 0].plot(hist["test_acc"],label=name)
        axs[1, 1].plot(hist["test_loss"],label=name)
    for ax in axs.flat:
        ax.set(xlabel='Communication rounds')
        ax.legend()
    plt.show()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss().to(args.device)

    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def get_dataset(args):
    if args.dataset == 'cifar10':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        else:
            data_dir = '../data/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    dataset            : {args.dataset}')
    print(f'    Num of users       : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Learning  Rate     : {args.lr}')
    print(f'    rho                : {args.rho}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}\n')
    return
