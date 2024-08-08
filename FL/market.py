import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from homo_encryption import enc_cos_similarity
import pickle


class Market(object):

    def __init__(self):
        print("市场初始化")
        self.blocks = []
        self.DAG = None
        self.node_to_block = {}
        self.block_to_node = {}
        self.node_cr = {}
        self.genesis_nodes = []

    def blocks_to_graph(self):
        self.DAG = nx.DiGraph()
        for block in self.blocks:
            node = block
            self.DAG.add_node(block)
            self.block_to_node[block] = node
            self.node_to_block[node] = block

            if block.header.StrongParent:
                self.DAG.add_edge(block, block.header.StrongParent,
                                  weight=block.header.StrongParentWeight)
            if block.header.WeakParent:
                self.DAG.add_edge(block, block.header.WeakParent,
                                  weight=block.header.WeakParentWeight)

    def add_block(self, block):
        self.blocks.append(block)
        self.DAG = self.blocks_to_graph()

    def show(self):
        pos = nx.spring_layout(self.DAG)  # 设置图的布局
        labels = {node: node.header.BlockId for node in self.DAG.nodes()}
        nx.draw(self.DAG, pos, with_labels=True, labels=labels,
                node_size=2000, font_size=10, font_color='black', arrows=True)
        nx.draw_networkx_edges(self.DAG, pos, width=1)
        plt.show()

    def update_cr(self, snode, bid):
        for node in self.DAG.nodes():
            if node.header.BlockId == snode.header.BlockId:
                node.CumulativeReward += bid
                if node in list(self.node_cr):
                    self.node_cr[node].append(node.CumulativeReward)
                else:
                    self.node_cr.update({node: [node.CumulativeReward]})
                break

    def cos_map(self, client):
        cos_map = []
        print("executing cos_map:")
        for block in self.blocks:
            print("block{}gradient:{} and client{} gradient{}".format(
                block.header.BlockId, block.content, client.id, client.updated))
            cos = enc_cos_similarity(block.content, client.updated)
            cos_map.append((block.header.BlockId, cos))
        for item in cos_map:
            print(item)

    def show_edge(self):
        for node in self.DAG.nodes():
            print("node{}'s 后继节点:{}".format(
                node.header.BlockId, list(self.DAG.predecessors(node))))
            print("node{}'s 前驱节点:{}".format(node.header.BlockId,
                                            list(self.DAG.successors(node))))
            print("*"*30)

    def upload_graph(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.DAG, f)

    def download_graph(self, path):
        with open(path, 'rb') as f:
            self.DAG = pickle.load(f)
