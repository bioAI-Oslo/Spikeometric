import numpy as np
import torch
import math

class NormalWeights:
    def __init__(self, n_neurons, mu, sigma):
        self.n_neurons = n_neurons
        self.mu = mu
        self.sigma = sigma
        self.ref_scale = 10
        self.abs_ref_scale = 3
        self.spike_scale = 5
        self.abs_ref_strength = -100
        self.rel_ref_strength = -30
        self.alpha = 0.2
        self.beta = 0.5

    def build_W0(self, rng):
        half_n = int(self.n_neurons / 2)
        W0 = self._generate_connectivity_matrix((half_n, half_n), rng)
        W0 = self._dales_law(W0)
        return W0

    def build_W(self, W0):
        # W_decreasing = self._construct_connectivity_filters(W0)
        W_decreasing = self.construct_W(W0)
        edge_index = torch.nonzero(W_decreasing[:, :, 0]).T
        W = W_decreasing.flip(dims=(2,))
        idx = torch.nonzero(W[:, :, -1])
        flattened_W = W[idx[:, 0], idx[:, 1]]

        return flattened_W, edge_index

    def _dales_law(self, W0):
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        W0 = torch.concat((W0, W0), 1)
        return W0

    def _generate_connectivity_matrix(self, shape, rng):
        W0 = torch.normal(self.mu, self.sigma, shape, generator=rng) / math.sqrt(self.n_neurons)
        W0.fill_diagonal_(0)
        return W0
    
    def _construct_connectivity_filters(self, W0):
        W = torch.zeros((W0.shape[0], W0.shape[1], self.ref_scale))
        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                if i == j:
                    W[i, j, : self.abs_ref_scale] = self.abs_ref_strength
                    abs_ref = torch.arange(self.abs_ref_scale, self.ref_scale)
                    W[i, j, self.abs_ref_scale : self.ref_scale] = self.rel_ref_strength * torch.exp(-0.5 * (abs_ref + self.abs_ref_scale + 1))
                else:
                    W[i, j, torch.arange(self.spike_scale)] = W0[i, j] * torch.exp(
                        -self.alpha * torch.arange(self.spike_scale)
                    )
        return W

    def construct_W(self, W0):
        # Construct W without loops?
        diagonal_identity = torch.eye(W0.shape[0]).unsqueeze(2)
        scalars = torch.ones(self.ref_scale)
        scalars[: self.abs_ref_scale] = self.abs_ref_strength
        abs_ref = torch.arange(self.abs_ref_scale, self.ref_scale)
        scalars[abs_ref] = self.rel_ref_strength * torch.exp(-self.beta * (abs_ref + self.abs_ref_scale + 1))
        # scalars[abs_ref] = self.rel_ref_strength * torch.exp(-self.beta * (abs_ref - self.abs_ref_scale))
        W = diagonal_identity @ scalars.unsqueeze(0)
        W[:, :, :self.spike_scale] += W0.unsqueeze(2) @ torch.exp(-self.alpha * torch.arange(self.spike_scale)).unsqueeze(0)
        return W


