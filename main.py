import argparse
from FL.market import Market
from FL.FLclient import clientDAG  # type: ignore
from FL.models import CNN, DNN, TextCNN
import math
from result_handler import Result_handler
import torchvision  # type: ignore
# import numpy as np
from message import *
from homo_encryption import *
import torch  # type: ignore
# import sys
import json
from FL.data_utils import *


# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32
result_handler = Result_handler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_clients(n, model, dataset):
    Clients = []
    for i in range(n):
        client = clientDAG(id=i, model=model, dataset=dataset)
        Clients.append(client)
    return Clients


def run(args):
    print("==========================INITIALIZATION========================")
    print("Starting Time:", datetime.now())
    MAX_ROUND = args.round
    n_clients = args.n_clients
    model_input = args.model
    dataset = args.dataset
    attack_ratio = args.attack_ratio
    attack_type = args.attack_type
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "CNN":
        if dataset == "MNIST":
            model_input = CNN(
                in_features=1, num_classes=10, dim=1024).to(args.device)
        elif dataset == "Cifar10":
            model_input = CNN(
                in_features=3, num_classes=10, dim=1600).to(args.device)
        elif dataset == "Cifar100":
            model_input = CNN(
                in_features=3, num_classes=100, dim=1600).to(args.device)
        elif dataset == "AG_news":
            model_input = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                  num_classes=args.num_classes).to(args.device)
    elif args.model == "DNN":
        if dataset == "MNIST":
            args.model = DNN(
                1*28*28, 100, num_classes=10).to(args.device)
        elif dataset == "Cifar10":
            args.model = DNN(
                3*32*32, 100, num_classes=10).to(args.device)
        elif dataset == "Cifar10":
            args.model = DNN(
                3*32*32, 100, num_classes=100).to(args.device)
    elif args.model == "ResNet":
        model_input = torchvision.models.resnet18(
            pretrained=False, num_classes=args.num_classes)
    Total_Clients = set_clients(n_clients, model_input, dataset)
    malicious_client = math.floor(len(Total_Clients)*args.attack_ratio)

    for i in range(malicious_client):
        if args.attack_type == "signrandom":
            Total_Clients[i].status = "signrandom"
        elif args.attack_type == "rescaling":
            Total_Clients[i].status = "rescaling"
        elif args.attack_type == "labelflip":
            Total_Clients[i].status = "labelflip"

    if args.walk == "greedy":
        walking_strategy = "greedy"
    elif args.walk == "random":
        walking_strategy = "random"
    elif args.walk == "performance":
        walking_strategy = "performance"
    elif args.walk == "cost":
        walking_strategy = "cost"

    print("Number of Clients:", len(Total_Clients))
    print("Number of Malicious Clients:", malicious_client)
    print("Training Round:", MAX_ROUND)
    print("Walking_strategy:", walking_strategy)
    print("Device:", device)
    print("Aggregation Method:", args.agg)
    print("Dataset:", dataset)
    print("Attack Ratio:", attack_ratio)
    print("Attack Type:", args.attack_type)
    print("Heterogenity:", args.noniid)
    market = Market()

    print("============================TRAINING==========================")
    for client in Total_Clients:
        if client.status == "signrandom":
            client.sign_randomnizing_train()
        elif client.status == "rescaling":
            client.rescaling_train()
        elif client.status == "labelflip":
            client.label_flip_train([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [
                3, 2, 1, 4, 6, 5, 7, 9, 8, 0])
        else:
            client.train()
        acc, num, auc = client.test_metrics()
        losses, num = client.train_metrics()
        accuracy = acc/num
        result_handler.clients_acc.update({client.id: [accuracy]})
        result_handler.clients_loss.update({client.id: [losses/num]})
        header = Header(client.id,
                        client.id+1000*0, datetime.now())
        metadata = Metadata(client.updated, client.model)
        block = Block(header, client.train_samples,
                      metadata)
        block.header.train_time_cost = client.train_time_cost
        # print("current block has training time cost:{}".format(
        # block.header.train_time_cost))
        market.blocks.append(block)
        market.genesis_nodes.append(block)
        market.blocks_to_graph()
    for round in range(MAX_ROUND):
        print(
            "============================Round {}==========================".format(round))
        for i in range(n_clients):
            current_client = Total_Clients[i]
            final_res, selected_blocks, selected_cos = current_client.greedy_select(
                market, walking_strategy, round)
            for res in final_res:
                crit_list = res[2]
                result_handler.crit2step.append(
                    [current_client.id, round, crit_list])
            path1 = final_res[0][1]
            path2 = final_res[1][1]
            final_path = path_merge(path1, path2)
            if args.agg == "similar":
                current_client.aggregate2blocks_similar(
                    selected_blocks[0], selected_blocks[1], selected_cos)
            if args.agg == "vanilla":
                current_client.aggregate2blocks(
                    selected_blocks[0], selected_blocks[1])
            if current_client.status == "signrandom":
                current_client.sign_randomnizing_train()
            elif current_client.status == "rescaling":
                current_client.rescaling_train()
            elif current_client.status == "labelflip":
                current_client.label_flip_train([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [
                    3, 2, 1, 4, 6, 5, 7, 9, 8, 0])
            else:
                current_client.train()
            acc, num, auc = current_client.test_metrics()
            result_handler.clients_acc[current_client.id].append(acc/num)
            losses, num = current_client.train_metrics()
            result_handler.clients_loss[current_client.id].append(losses/num)
            header = Header(current_client.id,
                            current_client.id+1000*round, datetime.now())
            metadata = Metadata(current_client.updated, current_client.model)
            block = Block(header, client.train_samples,
                          metadata)
            block.header.WeakParent = selected_blocks[0]
            block.header.StrongParent = selected_blocks[1]
            block.header.WeakParentWeight = 1/selected_cos[0]
            block.header.StrongParentWeight = 1/selected_cos[1]
            block.header.train_time_cost = current_client.train_time_cost
            market.blocks.append(block)
            market.blocks_to_graph()
            for item in final_path:
                node = list(item)[0]
                bid = item[node][1]
                market.update_cr(node, bid)
                client.outpay += bid
            for node in market.DAG.nodes():
                if node.header.BlockId in list(result_handler.round2cr):
                    result_handler.round2cr[node.header.BlockId].append(
                        [round, node.CumulativeReward])
                else:
                    result_handler.round2cr.update(
                        {node.header.BlockId: [round, node.CumulativeReward]})
            cos_path = []
            for item in final_path:
                block = list(item)[0]
                crit_list = item[block][2]
                cos = enc_cos_similarity(
                    block.metadata.content, current_client.updated)
                cos_path.append((block.header.BlockId, cos))

    cost_tot = 0
    # cost_tot_list = []
    acc_tot = []
    loss_tot = []
    for client in Total_Clients:
        acc = result_handler.clients_acc[client.id]
        acc_tot.append(acc)
        loss = result_handler.clients_loss[client.id]
        loss_tot.append(loss)
        cost_tot += client.outpay
    # cost_tot_list.append(cost_tot)
    acc_mean = mean_list(acc_tot)
    loss_mean = mean_list(loss_tot)
    '''
    with open(str(args.agg)+str(n_clients)+'_acc_list_'+dataset+'_'+str(attack_type)+str(attack_ratio)+str(walking_strategy)+'.json', 'w') as file:
        json.dump(acc_mean, file)
    with open(str(args.agg)+str(n_clients)+'_loss_list_'+dataset+'_'+str(attack_type)+str(attack_ratio)+str(walking_strategy)+'.json', 'w') as file:
        json.dump(loss_mean, file)
    with open(str(args.agg)+str(n_clients)+'_cost_list_'+dataset+'_'+str(attack_type)+str(attack_ratio)+str(walking_strategy)+'.json', 'w') as file:
        json.dump(cost_tot_list, file)
    '''
    print("============================RESULTS==========================")

    print("Averaged Accuracy", acc_mean)
    print("Averaged Loss", loss_mean)
    print("End Time:", datetime.now())
    print("Run {} clients {} round, by {} walking strategy".format(
        n_clients, MAX_ROUND, walking_strategy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAGDFL')
    parser.add_argument('-n_clients', type=int,
                        help='NUMBER OF CLIENTS', default=10)
    parser.add_argument('-model', type=str, help='MODEL', default='FedAvgCNN')
    parser.add_argument('-dataset', type=str, help='DATASET')
    parser.add_argument('-attack_type', type=str,
                        help='ATTACK TYPE', default='rescaling')
    parser.add_argument('-attack_ratio', type=float,
                        help='ATTACK DENSITY', default=0)
    parser.add_argument('-round', type=int, help='TRAINING ROUND', default=5)
    parser.add_argument("-walk", type=str,
                        help="WALKING STRATEGY", default="greedy")
    parser.add_argument(
        "-agg", type=str, help="AGGREGATION METHOD", default="similar")
    args = parser.parse_args()
    run(args)

'''
def save_metrics(Total_Clients):
    acc_tot = []
    loss_tot = []
    for client in Total_Clients:
        print("用户每轮准确率变化:")
        result_handler.show_acc(client)
        result_handler.show_loss(client)
        acc = result_handler.clients_acc[client.id]
        acc_tot.append(acc)
        loss = result_handler.clients_loss[client.id]
        loss_tot.append(loss)
    acc_mean = mean_list(acc_tot)
    with open('acc_list.json', 'w') as file:
        json.dump(acc_mean, file)
    loss_mean = mean_list(loss_tot)
    with open('loss_list.json', 'w') as file:
        json.dump(loss_mean, file)
'''
