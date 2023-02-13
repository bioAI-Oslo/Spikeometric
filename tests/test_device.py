import torch
from torch_geometric.loader import DataLoader

def test_all_tensors_on_same_device(bernoulli_glm, data_with_stimulus_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bernoulli_glm.to(device)
    data_with_stimulus_mask.to(device)
    for parameter in bernoulli_glm.state_dict():
        assert bernoulli_glm.state_dict()[parameter].is_cuda == torch.cuda.is_available()
    assert data_with_stimulus_mask.W0.is_cuda == torch.cuda.is_available()
    assert data_with_stimulus_mask.edge_index.is_cuda == torch.cuda.is_available()
    assert data_with_stimulus_mask.stimulus_mask.is_cuda == torch.cuda.is_available()

def test_simulates_on_gpu_if_available(bernoulli_glm, data_with_stimulus_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bernoulli_glm.to(device)
    data_with_stimulus_mask.to(device)
    spikes = bernoulli_glm.simulate(data_with_stimulus_mask, 10, verbose=False)
    assert spikes.is_cuda == torch.cuda.is_available()
