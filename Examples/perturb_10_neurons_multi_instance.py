import numpy as np
import torch
from numpy.random import default_rng
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, freeze_support, current_process, RLock
from functools import partial
import pathlib
import sys
sys.path.append('..')

from generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    simulate_torch,
    construct_input_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    dales_law_transform,
    simulate_torch,
    # simulate
)

def construct(params, rng=None):
    rng = default_rng() if rng is None else rng
    # set stim
    drive1 = generate_poisson_stim_times(
        params['drive1_period'],
        params['drive1_isi_min'],
        params['drive1_isi_max'],
        params['n_time_step'],
        rng=rng
    )

    drive2 = generate_poisson_stim_times(
       params['drive2_period'],
       params['drive2_isi_min'],
       params['drive2_isi_max'],
       params['n_time_step'],
       rng=rng
    )
    stimulus = np.concatenate((drive1, drive2), 0)
    W_0 = construct_connectivity_matrix(params)
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    # here the drive strength is added to the connectivity tensor, make sure
    # the construction is concecutive wrt to the stimulus indices
    W = construct_input_filters(
        W,
        {i: params['drive1_strength'] for i in excit_idx[:params['drive1_n_stim']]},
        params['drive1_scale'])
    W = construct_input_filters(
        W,
        {i: params['drive2_strength'] for i in excit_idx[:params['drive2_n_stim']]},
        params['drive2_scale'])

    return W, W_0, stimulus, excit_idx, inhib_idx

def _multiprocess_simulate(i, **kwargs):
    rng = torch.Generator(device=kwargs['device'])
    rng.manual_seed(i)
    kwargs['rng'] = rng # seed each process
    # nice progressbar for each process
    from multiprocessing import current_process
    if kwargs['pbar'] is not None:
        current = current_process()
        pos = current._identity[0] - 1

        kwargs['pbar'] = partial(tqdm, position=pos)
    return simulate_torch(**kwargs)

# def _multiprocess_simulate(i, **kwargs):
#     kwargs['rng'] = default_rng(i) # seed each process
#     # nice progressbar for each process
#     from multiprocessing import current_process
#     if kwargs['pbar'] is not None:
#         current = current_process()
#         pos = current._identity[0] - 1
#
#         kwargs['pbar'] = partial(tqdm, position=pos)
#     return simulate(**kwargs)

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/')
    data_path.mkdir(parents=True, exist_ok=True)

    params = {
        'const': 5,
        'n_neurons': 10,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'drive1_n_stim': 5,
        'drive1_scale': 2,
        'drive1_strength': 5,
        'drive1_period': 50,
        'drive1_isi_min': 10,
        'drive1_isi_max': 200,
        'drive2_n_stim': 5,
        'drive2_scale': 10,
        'drive2_strength': 5,
        'drive2_period': 100,
        'drive2_isi_min': 20,
        'drive2_isi_max': 200,
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': 5
        },
        'n_time_step': int(5e6),
        'seed': 12345,
    }
    # torch
    num_cores = 10
    # cpu
    # ss = SeedSequence(params['seed'])
    # num_cores = multiprocessing.cpu_count()
    # child_seeds = ss.spawn(num_cores)

    rng = default_rng(params['seed'])

    fname =  f'n10_ss5_s5'

    W, W_0, stimulus, excit_idx, inhib_idx = construct(params, rng=rng)
    pool = Pool(
        initializer=tqdm.set_lock,
        initargs=(RLock(),),
        processes=num_cores
    )
    with pool as p:
        res = p.map(
            partial(
                _multiprocess_simulate,
                W=W,
                W_0=W_0,
                inputs=stimulus,
                params=params,
                pbar=True,
                device='cuda'
            ),
            range(params['seed'],params['seed']+num_cores))

    # cpu
    # freeze_support()
    # pool = Pool(
    #     initializer=tqdm.set_lock,
    #     initargs=(RLock(),),
    #     processes=num_cores
    # )
    # with pool as p:
    #     res = p.map(
    #         partial(
    #             _multiprocess_simulate,
    #             W=W,
    #             W_0=W_0,
    #             inputs=stimulus,
    #             params=params,
    #             pbar=True
    #         ),
    #         child_seeds)

    np.savez(
        data_path / fname,
        data=res,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )
