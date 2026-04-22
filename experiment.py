import argparse
import logging
import os
import pickle
import random
import sys
import warnings

import dsrl  # noqa: F401  - registers DSRL envs with gymnasium
import gym  # noqa: F401  - kept for backward compatibility with D4RL pickles
import gymnasium  # noqa: F401
import numpy as np
import torch
import wandb
from tqdm import tqdm

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode_rtg,
    evaluate_episode_rtg_prom2_parallel,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer
from utils.set_env import set_env

original_stdout = sys.stdout

config = {
    # gpt-dt dict(n_layer=3, n_head=1, embed_dim=128)
    'gpt-mini':     dict(n_layer=6, n_head=6, embed_dim=192),
    'gpt-micro':    dict(n_layer=4, n_head=4, embed_dim=128),
    'gpt-nano':     dict(n_layer=3, n_head=3, embed_dim=48),
    'gpt2':         dict(n_layer=12, n_head=12, embed_dim=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, embed_dim=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, embed_dim=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, embed_dim=1600), # 1558M params
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)
warnings.filterwarnings('ignore')


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def seed_all(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def experiment(
        exp_prefix,
        variant,
):
    if variant['gpt'] == 'gpt-dt':
        pass
    elif variant['gpt'] == 'gpt-nano':
        variant.update(config['gpt-nano'])
    elif variant['gpt'] == 'gpt-micro':
        variant.update(config['gpt-micro'])
    elif variant['gpt'] == 'gpt-mini':
        variant.update(config['gpt-mini'])
    elif variant['gpt'] == 'gpt2':
        variant.update(config['gpt2'])
    elif variant['gpt'] == 'gpt2-medium':
        variant.update(config['gpt2-medium'])
    elif variant['gpt'] == 'gpt2-large':
        variant.update(config['gpt2-large'])
    elif variant['gpt'] == 'gpt2-xl':
        variant.update(config['gpt2-xl'])
    else:
        raise KeyError

    seed_all(variant['seed'])

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'


    exp_name = variant['exp_name']
    if not os.path.exists(os.getcwd()+f'/save'):
        os.makedirs(os.getcwd()+f'/save')
    if not os.path.exists(os.getcwd()+f'/save/{env_name}'):
        os.makedirs(os.getcwd()+f'/save/{env_name}')
    save_path = os.getcwd()+f'/save/{env_name}/{exp_name}_{dataset}'
    load_path = os.getcwd()+f'/save/{env_name}/{exp_name}_{dataset}.pt'


    env, max_ep_len, env_targets, scale, dataset_path = set_env(env_name, dataset)

    if env_name in ['hopper', 'halfcheetah', 'walker2d', 'ant', 'humanoid'] and env_name not in ['velocity']:
        is_gym_env = True
    else:
        is_gym_env = False

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_int = (env.action_space.high - env.action_space.low) / 2
    act_mid = env.action_space.low + act_int
    act_limit = (act_mid, act_int)

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns, costs = [], [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        if not is_gym_env:
            costs.append(path['costs'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    if not is_gym_env:
        print(f'Average cost: {np.mean(costs):.2f}, std: {np.std(costs):.2f}')
        print(f'Max cost: {np.max(costs):.2f}, min: {np.min(costs):.2f}')
    print('=' * 50)


    max_episode_returns = np.max(returns)
    min_episode_returns = np.min(returns)
    if not is_gym_env:
        env.max_episode_reward = np.max(returns)
        env.min_episode_reward = np.min(returns)
        env.set_target_cost(np.mean(costs))
        target_cost = np.mean(costs)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew, deterministic):
        def fn(model):
            returns, failures, costs, returns_nor, costs_nor = [], [], [], [], []
            returns_prom, costs_prom, failures_prom, returns_prom_nor, costs_prom_nor = \
                                                                                [], [], [], [], []
            list_dict = {'returns' : returns, 
                         'returns_nor': returns_nor, 
                         'costs' : costs, 
                         'costs_nor': costs_nor, 
                         'failures' : failures,
                         'return_prom': returns_prom,
                         'returns_prom_nor': returns_prom_nor, 
                         'costs_prom': costs_prom,
                         'costs_prom_nor': costs_prom_nor,
                         'failures_prom': failures_prom, 
                        }
            for _ in tqdm(range(num_eval_episodes)):
                sys.stdout = open('/dev/null', 'w')
                with torch.no_grad():
                    if is_gym_env:
                        ret, ret_nor, failure = evaluate_episode_rtg(
                                    env,
                                    state_dim,
                                    act_dim,
                                    model,
                                    deterministic=True,
                                    max_ep_len=max_ep_len,
                                    scale=scale,
                                    target_return=target_rew/scale,
                                    mode=mode,
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                                    max_return=max_episode_returns,
                                    min_return=min_episode_returns,
                        )
                        returns.append(ret); returns_nor.append(ret_nor); failures.append(failure)
                    else:
                        ret, cost, failure, ret_nor, cost_nor  = evaluate_episode_rtg(
                                    env,
                                    state_dim,
                                    act_dim,
                                    model,
                                    deterministic=True,
                                    max_ep_len=max_ep_len,
                                    scale=scale,
                                    target_return=target_rew/scale,
                                    mode=mode,
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                        )
                        returns.append(ret); costs.append(cost); failures.append(failure); returns_nor.append(ret_nor); costs_nor.append(cost_nor)
                    if variant['load']:
                        if is_gym_env:
                            ret_prom, ret_prom_nor, failure_prom = evaluate_episode_rtg_prom2_parallel(
                                        env,
                                        state_dim,
                                        act_dim,
                                        model,
                                        deterministic=deterministic,
                                        max_ep_len=max_ep_len,
                                        scale=scale,
                                        target_return=target_rew/scale,
                                        mode=mode,
                                        state_mean=state_mean,
                                        state_std=state_std,
                                        device=device,
                                        max_return=max_episode_returns,
                                        min_return=min_episode_returns,
                            )
                            returns_prom.append(ret_prom); returns_prom_nor.append(ret_prom_nor); failures_prom.append(failure_prom)
                        else:
                            ret_prom, cost_prom, failure_prom, ret_prom_nor, cost_prom_nor = \
                                        evaluate_episode_rtg_prom2_parallel(
                                        env,
                                        state_dim,
                                        act_dim,
                                        model,
                                        deterministic=deterministic,
                                        max_ep_len=max_ep_len,
                                        scale=scale,
                                        target_return=target_rew/scale,
                                        mode=mode,
                                        state_mean=state_mean,
                                        state_std=state_std,
                                        device=device,
                            )
                            returns_prom.append(ret_prom); costs_prom.append(cost_prom); failures_prom.append(failure_prom); returns_prom_nor.append(ret_prom_nor); costs_prom_nor.append(cost_prom_nor)
                sys.stdout = original_stdout

            return_dict = {}
            for key, val in list_dict.items():
                if len(val) != 0:
                    if np.mean(val) > 10:
                        return_dict[key+'_mean'] = round(np.mean(val), 2)
                    else:
                        return_dict[key+'_mean'] = round(np.mean(val), 3)
                    if not 'failure' in key:
                        return_dict[key+'_std'] = round(np.std(val), 2)
            
            return return_dict
        
        return fn

    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            act_limit=act_limit,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
    )

    if variant['load']:
        print('=' * 50 + ' Load Model')
        model = torch.load(load_path)
        print(load_path)
        model.eval()

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            deterministic=False,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(0.1*(s_hat - s)**2) + torch.mean((a_hat - a)**2) + torch.mean((r_hat - r)**2),
            eval_fns=[eval_episodes(tar, deterministic=False) for tar in env_targets],
            test_fns=[eval_episodes(tar, deterministic=True) for tar in env_targets],
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )

    if not variant['load']:
        best_returns = -100000
        for iter in range(variant['max_iters']):
            outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
            if log_to_wandb:
                wandb.log(outputs)
            if outputs['evaluation/returns_mean'] > best_returns:
                torch.save(model, save_path+f'.pt')
                print(save_path+f'.pt')
    else:
        with torch.no_grad():
            for iter in range(variant['n_seed']):
                seed_all(variant['seed']+iter)
                outputs = trainer.test_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt', type=str, default="gpt-dt")
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='expert')  # medium, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_seed', '-ns', type=int, default=1)
    parser.add_argument('--exp_name', '-exp', type=str, default='tmp')
    parser.add_argument('--load', '-l', action='store_true')
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
