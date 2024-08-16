import math
from sklearn.preprocessing import label_binarize
import pickle
import os
import numpy as np
import torch.nn as nn
import torch
import traceback
import datetime
import copy
from torch.utils.data import DataLoader
# from sklearn.preprocessing import label_binarize
# from sklearn import metrics
from FL.market import *
from homo_encryption import enc_cos_similarity
from .data_utils import *
from .Batch import Batch
from message import *
from collections import Counter


UNIT_PRICE = 10
UNIT_INCREASE = 100

# Partial code is from https://github.com/TsingZ0/PFLlib


class Client(object):

    def __init__(self, id, model, dataset, **kwargs):
        torch.manual_seed(0)

        self.id = id
        self.model = copy.deepcopy(model)
        self.dataset = dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.id = id  # integer
        self.num_classes = 10
        self.train_samples = None
        self.test_samples = None
        self.batch_size = 10
        self.learning_rate = 0.005
        self.local_epochs = 1
        self.train_time_cost = 0
        self.outpay = 0
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = False
        self.send_slow = False
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = False
        self.dp_sigma = 0.0

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.99
        )
        self.learning_rate_decay = False
        self.num_reference = 0
        self.status = "normal"
        self.outpay = 0

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        # print("Client {} loaded test data".format(self.id))
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        num_batches = len(testloaderfull)

        self.test_samples = num_batches * self.batch_size
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(),  # type: ignore
                                    classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        self.model.cpu()

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = float('nan')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def aggregate2blocks(self, bk1, bk2):
        uploaded_weights = []
        uploaded_weights.append(self.train_samples)
        uploaded_weights.append(bk1.weight)
        uploaded_weights.append(bk2.weight)
        tot_samples = self.train_samples + \
            bk1.weight+bk2.weight
        for i, w in enumerate(uploaded_weights):
            uploaded_weights[i] = w / tot_samples
        for client_param, obj1_param, obj2_param in zip(self.model.parameters(), bk1.metadata.model.parameters(), bk2.metadata.model.parameters()):
            # print("before param {}".format(client_param.data))
            updated_param = client_param * \
                uploaded_weights[0] + obj1_param * \
                uploaded_weights[1] + obj2_param * uploaded_weights[2]
            client_param.data.copy_(updated_param)

    def aggregate2blocks_similar(self, bk1, bk2, selected_cos):
        uploaded_weights = []
        uploaded_weights.append(1)
        uploaded_weights.append(selected_cos[0])
        uploaded_weights.append(selected_cos[1])
        tot_cos = math.exp(selected_cos[0])+math.exp(selected_cos[1])

        for i, w in enumerate(uploaded_weights):
            uploaded_weights[i] = (math.exp(w) / tot_cos)*0.5
        uploaded_weights[0] = 0.5
        for client_param, obj1_param, obj2_param in zip(self.model.parameters(), bk1.metadata.model.parameters(), bk2.metadata.model.parameters()):
            updated_param = client_param * \
                uploaded_weights[0] + obj1_param * \
                uploaded_weights[1] + obj2_param * uploaded_weights[2]
            client_param.data.copy_(updated_param)

    def greedy_walk(self, market, walking_way, round):
        tot_crit = 0
        crit_list = []
        cos_list = []
        market.genesis_nodes.sort(key=lambda x: enc_cos_similarity(
            x.metadata.content, self.updated), reverse=True)
        i = random.randint(0, 2)
        start_node = market.genesis_nodes[i]
        path = []
        current = start_node
        path.append({current: [0, 0, 0]})
        crit_list.append(0)
        MAX_STEPS = 10
        for i in range(MAX_STEPS):
            if len(list(market.DAG.predecessors(current))) != 0:
                crit_dict = {}
                neighbors = list(market.DAG.predecessors(current))
                neighbor_cos = []
                neighbor_filtered = []
                for neighbor in neighbors:
                    block = market.node_to_block[neighbor]
                    cos = enc_cos_similarity(
                        block.metadata.content, self.updated)
                    if cos <= 0:
                        pass
                    else:
                        neighbor_cos.append(cos)
                        neighbor_filtered.append(neighbor)
                        bid = UNIT_PRICE * self.train_samples*(math.exp(cos)-1)
                        perfromance = UNIT_INCREASE * (1-math.exp(-cos))
                        criterion = perfromance/bid
                        crit_dict.update(
                            {neighbor: [bid, perfromance, criterion]})
                if len(neighbor_cos) == 0:
                    pass
                else:
                    total_criteria = sum([item[-1]
                                          for item in crit_dict.values()])
                    probabilities = [
                        (item[-1]) / total_criteria for item in crit_dict.values()]

                    total_bid = sum([1/item[0] for item in crit_dict.values()])
                    probabilities_bid = [
                        ((1/(item[0])) / total_bid) for item in crit_dict.values()]
                    total_performance = sum([item[1]
                                            for item in crit_dict.values()])
                    probabilities_performance = [
                        (item[1]) / total_performance for item in crit_dict.values()]
                    if walking_way == "random":
                        print("random walking")
                        next_node = np.random.choice(neighbor_filtered)
                    elif walking_way == "greedy":
                        next_node = np.random.choice(
                            neighbor_filtered, p=probabilities)
                    elif walking_way == "cost":
                        print("cost walking")
                        next_node = np.random.choice(
                            neighbor_filtered, p=probabilities_bid)
                    elif walking_way == "performance":
                        print("performance walking")
                        next_node = np.random.choice(
                            neighbor_filtered, p=probabilities_performance)
                    cos = enc_cos_similarity(
                        next_node.metadata.content, self.updated)
                    cos_current = enc_cos_similarity(
                        current.metadata.content, self.updated)
                    if cos_current - cos >= 0.4:
                        break
                    cos = enc_cos_similarity(
                        next_node.metadata.content, self.updated)

                    min_neighbor = next_node
                    min_crit = crit_dict[min_neighbor][2]
                    current = min_neighbor
                    # 记录游走的节点的信息
                    path.append({current: crit_dict[current]})
                    tot_crit = tot_crit+min_crit
                    crit_list.append(tot_crit)
            elif len(list(market.DAG.predecessors(current))) == 0:
                break
        return current, path, crit_list

    def greedy_select(self, market, walking_way, round):
        selected_blocks = []
        selected_cos = []
        results = []
        print("client {} begins greedy random walk".format(self.id))
        for _ in range(20):
            current, path, crit_list = self.greedy_walk(
                market, walking_way, round)
            results.append([current, path, crit_list])
        end_nodes = [result[0] for result in results]
        end_node_counter = Counter(end_nodes)

        most_common = end_node_counter.most_common(2)
        final_res = []
        for node, count in most_common:
            node_results = [res for res in results if res[0] == node]
            min_cost_result = node_results[0]
            for res in node_results:
                if res[-1][-1] <= min_cost_result[-1][-1]:
                    min_cost_result = res
            final_res.append(min_cost_result)
        if len(final_res) == 1 and round == 0:
            final_res.append(results[0])
        if len(final_res) == 1:
            final_res.append(final_res[0])
        for res in final_res:
            block = res[0]
            path = res[1]
            cos = enc_cos_similarity(block.metadata.content, self.updated)
            selected_blocks.append(block)
            selected_cos.append(cos)
        return final_res, selected_blocks, selected_cos
