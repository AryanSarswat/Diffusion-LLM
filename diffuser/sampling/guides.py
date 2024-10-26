import torch
import torch.nn as nn
import pdb

class CustomGuide(nn.Module):
    def __init__(self, loss_fn, model):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model

    def forward(self, x, t):
        loss_values = self.loss_fn(x, self.model.observation_dim)
        return loss_values  # Shape: [batch_size]


    def gradients(self, x, cond, t):
        x = x.clone().detach().requires_grad_(True)
        # Compute per-sample loss
        loss_values = self.forward(x, t)  # Shape: [batch_size]
        # Compute gradients: sum loss_values to aggregate gradients for backprop
        grad = torch.autograd.grad(loss_values.sum(), x)[0]  # Shape: [batch_size, horizon, transition_dim]
        # Detach x from the computation graph
        x = x.detach()
        return loss_values.detach(), grad  # Both tensors are detached

class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad
