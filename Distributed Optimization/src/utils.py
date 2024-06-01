import torch
import numpy as np
import random
from torchvision import datasets, transforms
from sampling import iid_split, noniid_split
from torch.utils.data import Dataset
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


def servers_plot(servers, clients, frac, iid,labels):
    title = "| {} Clients | frac: {} | iid: {} |".format(clients, frac, iid)
    fig, axs = plt.subplots(2, 2, figsize=(30, 15))
    fig.suptitle(title, fontsize=36)
    axs[0, 0].set_title('Average train accuracy of all clients', fontsize=22)
    axs[0, 0].set_ylabel('Average Accuracy')
    axs[0, 1].set_title('Average training loss of clients in a round', fontsize=22)
    axs[0, 1].set_ylabel('Training loss')
    axs[1, 0].set_title('Average Test Accuracy of all clients', fontsize=22)
    axs[1, 0].set_ylabel('test_acc')
    axs[1, 1].set_title('Average Test Loss of all clients', fontsize=22)
    axs[1, 1].set_ylabel('test_loss')
    for i,server in enumerate(servers):
        hist = pd.DataFrame(server.history)
        name = labels[i]
        # axs[0, 0].plot(hist["train_acc"], label=name)
        axs[0, 1].plot(hist["avg_train_loss"], label=name)
        axs[1, 0].plot(hist["avg_test_acc"], label=name)
        axs[1, 1].plot(hist["avg_test_loss"], label=name)
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
    if args.verbose:
        print(f"\n | Download Dataset {args.dataset} |")
    data_dir = "../data/{}/".format(args.dataset)
    if args.dataset == 'cifar10':
        # apply_transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
        apply_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
    elif args.dataset == 'cifar100':
        # apply_transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
        apply_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform)
    elif args.dataset == 'mnist':
        apply_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        # apply_transform = transforms.Compose([transforms.Resize(28), transforms.CenterCrop(28), transforms.ToTensor()])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,transform=apply_transform)
    elif args.dataset == 'fmnist':
        # apply_transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
        apply_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,transform=apply_transform)
    if args.iid:
        if args.verbose:
            print("\n | Splitting Dataset |")
            print(f'| IID | # Imgs Per Client : {len(train_dataset)//args.num_users} |')
        user_groups = iid_split(train_dataset, args)
    else:
        if args.verbose:
            print("\n | Splitting Dataset |")
            print(f'| Non-IID | # of Shards : {args.shards} | Client # Imgs : {len(train_dataset)//args.num_users} |')
        user_groups = noniid_split(train_dataset, args)
    return train_dataset, test_dataset, user_groups


def exp_details(args):
    print(f"\n | Parameters details |")
    print(f'    Model               : {args.model}')
    print(f'    Optimizer           : {args.optimizer}')
    print(f'    Global Rounds       : {args.epochs}')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Dataset             : {args.dataset}')
    print(f'    Num of users        : {args.num_users}')
    print(f'    Fraction of users   : {args.frac}')
    print(f'    Learning  Rate      : {args.lr}')
    print(f'    Rho                 : {args.rho}')
    print(f'    Rho Policy          : {args.rho_policy}')
    print(f'    Local Epochs        : {args.local_ep}')
    print(f'    Local Epochs Fix    : {args.fixed}')
    print(f'    Local Batch size    : {args.local_bs}')
    print(f'    Local model loading : {args.loading}')
    print(f'    Eta                 : {args.eta}')
    print(f'    Efficiency type     : {args.efficiency}')
    print(f'    Random Seed         : {args.seed}')
    return
