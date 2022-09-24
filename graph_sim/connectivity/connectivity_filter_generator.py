import torch

class ConnectivityFilterGenerator:
    def __init__(self, n_neurons, w0_params, filter_params):
        self.n_neurons = n_neurons
        self.mu = w0_params.mean
        self.sigma = w0_params.std
        self.ref_scale = filter_params.ref_scale
        self.abs_ref_scale = filter_params.abs_ref_scale
        self.spike_scale = filter_params.spike_scale
        self.abs_ref_strength = filter_params.abs_ref_strength
        self.rel_ref_strength = filter_params.rel_ref_strength
        self.decay_offdiag = filter_params.decay_offdiag
        self.decay_diag = filter_params.decay_diag
        self.n_edges = None

    def build_W0(self, rng):
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        half_n = int(self.n_neurons / 2)
        half_W0 = self._generate_connectivity_matrix((half_n, half_n), rng)
        W0 = self._dales_law(half_W0)
        return W0

    def _dales_law(self, W0):
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        W0 = torch.concat((W0, W0), 1)
        return W0

    def _generate_connectivity_matrix(self, shape, rng):
        """Generates a connectivity matrix of a given shape from a normal distribution, with zeros on the diagonal"""
        W0 = torch.normal(self.mu, self.sigma, shape, generator=rng) / torch.sqrt(torch.tensor(self.n_neurons))
        W0.fill_diagonal_(0)
        return W0
    
    def build_W(self, W0):
        """Constructs a connectivity filter W from the weight matrix W0 and the filter parameters"""
        # Sets the diagonal elements of the connectivity filter
        diagonal_identity = torch.eye(W0.shape[0]).unsqueeze(2)

        # Sets self-influence during the absolute refractory period to -self.abs_ref_strength
        diag = torch.zeros(self.ref_scale)
        diag[:self.abs_ref_scale] = self.abs_ref_strength 
        rel_ref_idx = torch.arange(self.abs_ref_scale, self.ref_scale) # [3, 4, 5, 6, 7, 8, 9]

        # Sets self-influence during the relative refractory period to -self.rel_ref_strength * exp(-self.beta * (t+3))
        diag[rel_ref_idx] = self.rel_ref_strength * torch.exp(-self.decay_diag * (rel_ref_idx - self.abs_ref_scale))
        # diag[diag_idx] = self.rel_ref_strength * torch.exp(-self.beta * (diag_idx + self.abs_ref_scale + 1)) 

        # Sets the diagonal part of the connectivity filter (in decreasing time order)
        W = diagonal_identity @ diag.flip(dims=(0,)).unsqueeze(0)

        # Sets the off-diagonal elements of the connectivity filter based on W0 with decay self.alpha (in decreasing time order)
        offdiag_idx = torch.arange(self.spike_scale)
        offdiag = torch.exp(-self.decay_offdiag * offdiag_idx)
        W[:, :, -self.spike_scale:] += W0.unsqueeze(2) @ offdiag.flip(dims=(0,)).unsqueeze(0)
        edge_index = torch.nonzero(W[:, :, -1]).T
        flattened_W = W[edge_index[0], edge_index[1]]
        return flattened_W, edge_index


    def new_filter(self, p_sims, rng):
        """Prepare a new W and edge_index for p_sims parallel simulations"""
        Ws = []
        edge_indices = []
        for i in range(p_sims):
            W0 = self.build_W0(rng)
            W, edge_index = self.build_W(W0)
            Ws.append(W)
            edge_indices.append(edge_index)

        W = torch.concat(Ws, dim=0)

        edge_index = torch.concat(edge_indices, dim=1)
        shift = (torch.arange(p_sims)*self.n_neurons).repeat_interleave(W.shape[0] // p_sims, 0).repeat(2, 1)
        edge_index += shift

        return W, edge_index

