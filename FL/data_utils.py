from collections import defaultdict
import numpy as np
import os
import torch
from collections import defaultdict

# Partial code is from https://github.com/TsingZ0/PFLlib


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(
            './dataset', dataset, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        return train_data

    else:
        test_data_dir = os.path.join(
            './dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y)
                      for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y)
                     for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def flatten(grad_update):
    return torch.cat([grad_update.data.view(-1)])


def clip_gradient_update(grad_update, grad_clip):
    for param in grad_update:
        # param = torch.tensor(param, dtype=torch.float)
        update = torch.clamp(param, min=-grad_clip, max=grad_clip)
    return update


def compute_update_gradients(old_model, new_model):
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def cosine_xy(x, y):
    x = x.clone().detach()
    y = y.clone().detach()
    torch_cosine = torch.cosine_similarity(x, y, dim=0)
    return torch_cosine.item()


def mean_list(lists):
    return [sum(x) / len(x) for x in zip(*lists)]


def path_merge(path1, path2):
    path1_nodes = []
    for item in path1:
        node = list(item)[0]
        path1_nodes.append(node)
    for item in path2:
        node = list(item)[0]
        if node not in path1_nodes:
            path1.append(item)
    return path1


def filter_list(original_list):
    grouped = defaultdict(list)

    for item in original_list:
        grouped[item[0]].append(item[1])

    # 合并相同的第一个元素的列
    merged_list = [(key, sum(values)) for key, values in grouped.items()]

    return merged_list
