"""Download Safety Gymnasium (DSRL) expert datasets and serialize them as
pickles of trajectories.

Examples:
    python data/download_dsrl_datasets.py --env OfflinePointGoal1-v0
    python data/download_dsrl_datasets.py --download_all
"""

import argparse
import collections
import os
import pickle
import urllib

import dsrl  # noqa: F401  - registers DSRL envs with gymnasium
import gymnasium
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


set_dataset_path(
    os.environ.get('D4RL_DATASET_DIR', os.path.expanduser('~/workspace/.d4rl/datasets'))
)


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    return os.path.join(DATASET_PATH, dataset_name)


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


def main(name):
    version = name.split('-')[1]
    name = name.split('-')[0]
    proficiency = 'expert'
    env = gymnasium.make(f'{name}-{version}')

    data_dict = env.get_dataset()

    for key in ['observations', 'actions', 'rewards', 'costs', 'terminals', 'timeouts']:
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
    if data_dict['costs'].shape == (N_samples, 1):
        data_dict['costs'] = data_dict['costs'][:, 0]
    assert data_dict['costs'].shape == (N_samples,), (
        'Cost has wrong shape: %s' % str(data_dict['costs'].shape)
    )
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), (
        'Terminals has wrong shape: %s' % str(data_dict['rewards'].shape)
    )
    if data_dict['timeouts'].shape == (N_samples, 1):
        data_dict['timeouts'] = data_dict['timeouts'][:, 0]
    assert data_dict['timeouts'].shape == (N_samples,), (
        'Timeouts has wrong shape: %s' % str(data_dict['rewards'].shape)
    )
    dataset = data_dict

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = dataset['terminals'][i]
        final_timestep = dataset['timeouts'][i] if use_timeouts else False
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'costs',
                  'terminals', 'timeouts']:
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

    with open(f'{name}-{proficiency}-{version}.pkl', 'wb') as f:
        pickle.dump(paths, f)
    print(f'Data saved at {os.getcwd()}/{name}-{version}_{proficiency}')


ALL_DSRL_ENVS = [
    'OfflinePointGoal1-v0', 'OfflinePointGoal2-v0',
    'OfflinePointPush1-v0', 'OfflinePointPush2-v0',
    'OfflinePointButton1-v0', 'OfflinePointButton2-v0',
    'OfflineCarGoal1-v0', 'OfflineCarGoal2-v0',
    'OfflineCarPush1-v0', 'OfflineCarPush2-v0',
    'OfflineCarButton1-v0', 'OfflineCarButton2-v0',
    'OfflineHalfCheetahVelocity-v1', 'OfflineSwimmerVelocity-v1',
    'OfflineWalker2dVelocity-v1',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='OfflinePointGoal1-v0')
    parser.add_argument('--download_all', '-da', action='store_true')
    args = vars(parser.parse_args())

    if args['download_all']:
        for n in ALL_DSRL_ENVS:
            main(n)
    else:
        main(args['env'])
