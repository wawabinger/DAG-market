from datetime import datetime
import random


class Header:
    def __init__(self, publisher, block_id, time):
        self.publisher = publisher
        self.BlockId = block_id
        self.StrongParent = None
        self.StrongParentWeight = None
        self.WeakParent = None
        self.WeakParentWeight = None
        self.timestamp = time
        self.train_time_cost = 0


class Block:
    def __init__(self, header, train_samples, Metadata):
        self.header = header
        self.CumulativeReward = 0
        self.dataset = "Cifar10"
        self.model_name = "CNN"
        self.weight = train_samples
        self.metadata = Metadata

    def __repr__(self):
        return f"Block{self.header.BlockId}"


class Metadata:
    def __init__(self, content, model_params):
        self.content = content
        self.model = model_params


class Message:
    def __init__(self, block, sender, receiver, message_status):
        current_time = datetime.now()
        self.message_id = hash(str(block) + sender +
                               receiver + message_status + str(current_time))
        self.payload = block
        self.sender = sender
        self.receiver = receiver
        self.message_status = message_status
        # print("Message created ")

    def __repr__(self):
        return f"Message(id={self.message_id}, sender={self.sender}, receiver={self.receiver}, message_status={self.message_status})"
