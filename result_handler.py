import os


class Result_handler(object):

    def __init__(self):
        self.be_acc = []
        self.at_acc = []
        self.ag_sim = []
        self.clients_acc = {}
        self.clients_loss = {}
        self.block_bid = {}
        self.block_cr = {}
        self.time2bid = {}
        self.round2cr = {}
        self.crit2step = []
        self.reward2client = {}

    def show_acc(self, client):
        print("Client {} accuracy at each round: ".format(client.id))
        print(self.clients_acc[client.id])

    def show_loss(self, client):
        print("Client{} loss at each round:".format(client.id))
        print(self.clients_loss[client.id])
