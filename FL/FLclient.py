from .data_utils import *
from .privacy import *
import copy
import torch
import numpy as np
import time
from .clientbase import Client
from homo_encryption import enc_cos_similarity
from .Batch import *


class clientDAG(Client):

    def __init__(self, id, model, dataset):
        super().__init__(id, model, dataset)
        self.updated = None
        self.trainloader = self.load_train_data()
        num_batches = len(self.trainloader)
        self.train_samples = num_batches * self.batch_size
        print("Client {} initialized".format(self.id))

    def train(self):
        print("Client {} starts training\n".format(self.id))
        strat_time = time.time()
        trainloader = self.load_train_data()
        num_batches = len(trainloader)
        self.train_samples = num_batches * self.batch_size

        old_model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self.model.train()

        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer,
                              trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print(f"Layer: {name}, Gradient Updated: False")
        self.model.cpu()
        new_model = copy.deepcopy(self.model)
        updated_graidents = compute_update_gradients(old_model, new_model)
        updated_flattened = flatten(clip_gradient_update(
            updated_graidents, 0.01))
        self.updated = copy.deepcopy(updated_flattened)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost = time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)
        self.train_time_cost = time.time() - strat_time

    def rescaling_train(self):
        old_model = copy.deepcopy(self.model)
        self.model = self.model.to(self.device)
        self.train()
        print("Client {} launch rescaling attack\n".format(self.id))
        noise = 10
        for grad in self.model.parameters():
            grad.data *= (noise * 2) * torch.rand(size=grad.shape,
                                                  device=grad.device) - noise
            grad.data *= noise
        self.model.cpu()
        new_model = copy.deepcopy(self.model)
        updated_graidents = compute_update_gradients(old_model, new_model)
        updated_flattened = flatten(clip_gradient_update(
            updated_graidents, 0.01))
        self.updated = copy.deepcopy(updated_flattened)

    def sign_randomnizing_train(self):
        print("Client {} launch sign_randomnizing attack\n".format(self.id))
        old_model = copy.deepcopy(self.model)
        self.train()
        for grad in self.model.parameters():
            grad.data *= torch.tensor(np.random.choice(
                [-1.0, 1.0], size=grad.shape), dtype=torch.float)
        new_model = copy.deepcopy(self.model)
        updated_graidents = compute_update_gradients(old_model, new_model)
        updated_flattened = flatten(clip_gradient_update(
            updated_graidents, 0.01))
        self.updated = copy.deepcopy(updated_flattened)

    def label_flip_train(self, before_label, after_label):
        print("Client {} launch label_flip attack\n".format(self.id))
        old_model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self.model.train()
        self.model = self.model.to(self.device)
        trainloader = self.load_train_data()
        num_batches = len(trainloader)
        self.train_samples = num_batches * self.batch_size
        start_time = time.time()
        for e in range(self.local_epochs):
            for batch in trainloader:
                if isinstance(batch, Batch):
                    data, label = batch.text, batch.label
                    data = data.permute(1, 0)
                else:
                    data, label = batch[0], batch[1]

                for i, l in enumerate(label):
                    if l in before_label:
                        label[i] = after_label[before_label.index(l)]

                data, label = data.to(self.device), label.to(self.device)
                self.model.zero_grad()
                pred = self.model(data)
                loss = self.loss(pred, label)
                loss.backward()
                self.optimizer.step()
        self.model.cpu()
        new_model = copy.deepcopy(self.model)
        updated_graidents = compute_update_gradients(old_model, new_model)
        updated_flattened = flatten(clip_gradient_update(
            updated_graidents, 0.01))
        self.updated = copy.deepcopy(updated_flattened)

        self.train_time_cost = time.time() - start_time
