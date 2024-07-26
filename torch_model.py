import torch

import numpy as np
import torch.nn as nn
import pyro.contrib.gp as gp


from collections import OrderedDict
from torch.utils.data import DataLoader
from pyro.infer import TraceMeanField_ELBO


from utils import enable_dropout, disentangle_uncertainty


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropout_rate=0.5, batch_norm=True):
        super(MLP, self).__init__()
        if batch_norm:
            self.feat_extractor = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(input_dims, hidden_dims)),
                ("bn1", nn.BatchNorm1d(hidden_dims)),
                ("relu1", nn.ReLU()),
            ]))
        else:
            self.feat_extractor = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(input_dims, hidden_dims)),
                ("relu1", nn.ReLU(inplace=True)),
                ("dropout1", nn.Dropout(dropout_rate)),
            ]))
        self.output_layer = nn.Linear(hidden_dims, output_dims)

    def forward(self, input, analysis=False):
        if analysis:
            return self.feat_extractor(input)
        return self.output_layer(self.feat_extractor(input))


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def train(self, epochs, train_loader, optimizer):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


class Ensemble(BaseModel):
    def __init__(self, input_dims, hidden_dims, output_dims, lr, num_nets, use_cuda=True):
        super().__init__()
        self.net = nn.ModuleList(
            [MLP(input_dims, hidden_dims, output_dims) for _ in range(num_nets)])
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x, mean=True):
        outputs = []
        for model in self.net:
            outputs.append(model(x))
        if mean:
            return torch.mean(torch.stack(outputs), dim=0)
        else:
            return torch.stack(outputs).squeeze()

    def train(self, epochs, train_loader):
        criterion = torch.nn.BCEWithLogitsLoss()
        self.net.train()
        for i in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()
                # Output from model
                output = self.forward(data).view(-1)
                # Calc loss and backprop gradients
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print("[Epochs: %04d] loss: %.4f" %
                  (i + 1, total_loss / len(train_loader)))

    @torch.no_grad()
    def evaluate(self, test_x, num_forwards):
        self.net.eval()
        if self.use_cuda:
            test_x = test_x.cuda()
        test_loader = DataLoader(test_x, batch_size=128)
        ensemble_predictions = []
        for data in test_loader:
            ensemble_predictions.append(
                np.stack([nn.Sigmoid()(x).cpu().numpy() for x in self.forward(data, mean=False)]))
        ensemble_predictions = np.concatenate(ensemble_predictions, axis=1).T
        # ensemble_predictions
        mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
            ensemble_predictions)
        return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


class Deep_GP(BaseModel):
    def __init__(self, input_dims, hidden_dims, output_dims, lr, num_data, train_loader, use_cuda=True):
        super().__init__()
        self.net = self.__build_gpmodule(
            input_dims, hidden_dims, output_dims, num_data, train_loader)
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def __build_gpmodule(self, input_dims, hidden_dims, output_dims, num_data, train_loader):
        extractor = MLP(input_dims, hidden_dims, 10)
        kernel = gp.kernels.Matern52(
            input_dim=10, lengthscale=torch.ones(10))
        deep_kernel = gp.kernels.Warping(kernel, iwarping_fn=extractor)
        likelihood = gp.likelihoods.Binary()
        latent_shape = torch.Size([])
        batches = []
        for i, (data, _) in enumerate(train_loader):
            batches.append(data)
            if i >= ((70 - 1) // 1024):
                break
        Xu = torch.cat(batches)[:70].clone()
        gpmodule = gp.models.VariationalSparseGP(
            X=Xu,
            y=None,
            kernel=deep_kernel,
            Xu=Xu,
            likelihood=likelihood,
            latent_shape=latent_shape,  # output dimension, here == 1
            num_data=num_data,
            whiten=True,
            jitter=2e-6,
        )

        return gpmodule

    def train(self, epochs, train_loader):
        elbo = TraceMeanField_ELBO()
        criterion = elbo.differentiable_loss
        for i in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                self.net.set_data(data, target)
                self.optimizer.zero_grad()
                loss = criterion(self.net.model, self.net.guide)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print("[Epochs: %04d] loss: %.4f" %
                  (i + 1, total_loss / len(train_loader)))

    @torch.no_grad()
    def evaluate(self, test_x, num_forwards):
        self.net.eval()
        if self.cuda:
            test_x = test_x.cuda()

        # get prediction of GP model on data
        f_loc, f_var = self.net(test_x)
        predictions = []
        for _ in range(num_forwards):
            predictions.append(self.net.likelihood(f_loc, f_var))
        predictions = torch.stack(predictions).cpu().numpy().T
        mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
            predictions)
        return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


class MCDropout(BaseModel):
    def __init__(self, input_dims, hidden_dims, output_dims, lr, rate, use_cuda=True):
        super().__init__()
        self.net = MLP(input_dims, hidden_dims, output_dims,
                       dropout_rate=rate, batch_norm=False)
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def train(self, epochs, train_loader):
        criterion = torch.nn.BCEWithLogitsLoss()
        self.net.train()
        for i in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()
                # Output from model
                output = self.net(data).view(-1)
                # Calc loss and backprop gradients
                loss = criterion(output, target)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
            print("[Epochs: %04d] loss: %.4f" %
                  (i + 1, loss / len(train_loader)))

    @torch.no_grad()
    def evaluate(self, test_x, num_forwards):
        self.net.eval()
        enable_dropout(self.net)
        dropout_predictions = []
        if self.use_cuda:
            test_x = test_x.cuda()
        for _ in range(num_forwards):
            pred_y = nn.Sigmoid()(self.net(test_x))
            dropout_predictions.append(pred_y.cpu().numpy())
        dropout_predictions = np.concatenate(dropout_predictions, axis=1)
        mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
            dropout_predictions)
        return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty
