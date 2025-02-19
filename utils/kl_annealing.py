import torch.nn as nn
import math
import torch

class KLAnnealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, k=0.0025, b=6.25):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            k,b (float): parameters that control the rate of weight change
        """

        self.current_step = 0
        self.bce = nn.BCELoss()
        self.total_steps = total_steps
        self.k = k
        self.b = b

    def __call__(self, x, y, logvar, mu):
        """
        Args:
            x (torch.tensor): model output
            y (torch.tensor): target
            mu (torch.Tensor): latent space mu
            logvar (torch.Tensor): latent space log variance
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        bce_loss = self.bce(x, y)
        out = kld * self._slope() + bce_loss
        return out

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        return

    def _slope(self):
        exponent = (-self.k*self.current_step + self.b)
        y = 1 / (1 + math.exp(exponent))
        return y