import numpy as np

def iid_split(dataset, args):
    num_items = int(len(dataset)/args.num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def noniid_split(dataset, args):
    num_shards = (args.shards*args.num_users)
    num_imgs = len(dataset)//(num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_users)}
    total = num_shards*num_imgs
    idxs = np.arange(total)
    labels = np.array(dataset.targets)[:total]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(args.num_users):
        rand_set = set(np.random.choice(idx_shard, args.shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
