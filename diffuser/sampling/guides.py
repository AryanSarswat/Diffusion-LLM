import torch
import torch.nn as nn
import pdb


class CustomGuide(nn.Module):
    def __init__(self, loss_fn, model,normalizer):
        """
        Initializes the CustomGuide.

        Args:
            loss_fn (callable or str): The loss function to use. It can be a callable
                                       that takes (x, obs_dim) and returns a tensor of shape [batch_size],
                                       or a string containing the function definition.
            model (torch.nn.Module): The diffusion model to extract observation dimensions.
        """
        super().__init__()
        self.model = model
        self.normalizer=normalizer

        if isinstance(loss_fn, str):
            # Dynamically compile the loss function from a string
            local_vars = {}
            exec(loss_fn, {}, local_vars)
            if 'loss_fn' not in local_vars:
                raise ValueError("The loss function string must define a function named 'loss_fn'")
            self.loss_fn = local_vars['loss_fn']
            if not callable(self.loss_fn):
                raise ValueError("The compiled loss_fn must be callable")
        elif callable(loss_fn):
            self.loss_fn = loss_fn
        else:
            raise TypeError("loss_fn must be either a callable or a string containing the function definition")

    def forward(self, x, t):
        """
        Computes the loss values for each trajectory.

        Args:
            x (torch.Tensor): Tensor of shape [batch_size, horizon, transition_dim].
            t (torch.Tensor): Tensor representing timesteps (unused in this loss function).

        Returns:
            torch.Tensor: Tensor of shape [batch_size], loss values per trajectory.
        """
        loss_values = self.loss_fn(x, self.model.observation_dim,self.model.action_dim,self.normalizer)
        return loss_values  # Shape: [batch_size]

    def gradients(self, x, cond, t):
        """
        Computes gradients of the loss with respect to x.

        Args:
            x (torch.Tensor): Tensor of shape [batch_size, horizon, transition_dim].
            cond (dict): Conditioning information (unused in this method).
            t (torch.Tensor): Tensor representing timesteps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing loss_values and gradients,
                                               both detached from the computation graph.
        """
        x.requires_grad_(True)
        loss_values = self.forward(x, t)
        #gok=torch.autograd.gradcheck(lambda x: self.forward(x.double(), t), x.double(), eps=1e-6, atol=1e-4, rtol=1e-3)
        grad = torch.autograd.grad(loss_values.sum(), x,retain_graph=True)[0]
        grad = torch.clamp(grad, min=-1.0, max=1.0)
        #x = x.detach()
        #print("GRAD: ",grad)
        """ print(f"Loss mean: {loss_values.mean().item()}")
        print(f"Non-zero gradients: {(grad != 0).sum().item()}")
        print(f"Gradient mean: {grad.abs().mean().item()}") """
        return loss_values.detach(), grad.detach()

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
