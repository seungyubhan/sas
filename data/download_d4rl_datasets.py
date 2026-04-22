"""Download a D4RL MuJoCo dataset and serialize it as a pickle of trajectories.

Example:
    python data/download_d4rl_datasets.py --env Hopper-v3 --proficiency medium
"""

import argparse
import collections
import os
import pickle
import urllib.request as urllib

import gym
import h5py
import numpy as np
from tqdm import tqdm


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(os.environ.get('D4RL_DATASET_DIR', os.path.expanduser('~/.d4rl/datasets')))


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    return os.path.join(DATASET_PATH, dataset_name)


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


def main(args):
    env_name = args['env']
    proficiency = args['proficiency']

    name = env_name.split('-')[0]
    env = gym.make(name + '-v4')

    h5path = download_dataset_from_url(
        f"http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"
        f"{name.lower()}_{proficiency}-v2.hdf5"
    )

    env = gym.make(name)
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:
                data_dict[k] = dataset_file[k][:]
            except ValueError:
                data_dict[k] = dataset_file[k][()]

    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key

    N_samples = data_dict['observations'].shape[0]
    if env.observation_space.shape is not None:
        assert data_dict['observations'].shape[1:] == env.observation_space.shape, (
            'Observation shape does not match env: %s vs %s'
            % (str(data_dict['observations'].shape[1:]), str(env.observation_space.shape))
        )
    assert data_dict['actions'].shape[1:] == env.action_space.shape, (
        'Action shape does not match env: %s vs %s'
        % (str(data_dict['actions'].shape[1:]), str(env.action_space.shape))
    )
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), (
        'Reward has wrong shape: %s' % str(data_dict['rewards'].shape)
    )
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), (
        'Terminals has wrong shape: %s' % str(data_dict['rewards'].shape)
    )
    dataset = data_dict

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000 - 1)
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {k: np.array(v) for k, v in data_.items()}
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(
        f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, '
        f'max = {np.max(returns)}, min = {np.min(returns)}'
    )

    with open(f'{name}-{proficiency}-v2.pkl', 'wb') as f:
        pickle.dump(paths, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v3')
    parser.add_argument('--proficiency', '-p', type=str, default='expert')
    args = vars(parser.parse_args())
    main(args)
