import numpy as np
import torch
import random

from collections import deque


def normalize_returns(returns, max_return, min_return):
    normalized_returns = (returns - min_return) / (max_return - min_return)
    return round(normalized_returns, 3)


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        max_return=1,
        min_return=0,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        target_cost=0.0001,
        deterministic=True,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        max_return=1,
        min_return=0,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    is_gym_env_v3 = env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3']
    if is_gym_env_v3:
        state = env.reset()
    else:
        state = env.reset()[0]
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length, cost_epi = 0, 0, []
    reward_que = deque()
    failure = False
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            deterministic,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if is_gym_env_v3:
            state, reward, done, info = env.step(action)
        else:
            state, reward, terminal, timeout, info = env.step(action)
            done = terminal or timeout

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        if 'cost' in info.keys():
            cost_epi.append(abs(info['cost']))

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=0)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if is_gym_env_v3:
            reward_que.append(reward)
            if len(reward_que) > 40:
                _ = reward_que.popleft()

            if episode_length > 70:
                if sum(reward_que) < 0:
                    break

        if done:
            if 'cost' in info.keys():
                episode_cost = sum(cost_epi)
                cost_mean = episode_cost / len(cost_epi)
                if cost_mean > target_cost:
                    failure = True
                cost_epi = []
            if not is_gym_env_v3:
                if terminal:
                    failure = True
            break

    if is_gym_env_v3 and t+1 < max_ep_len:
        failure = True

    if 'cost' in info.keys():
        episode_return_nor, episode_cost_nor = env.get_normalized_score(episode_return, episode_cost)
        return episode_return, episode_cost, failure, episode_return_nor, episode_cost_nor
    else:
        episode_return_nor = normalize_returns(episode_return, max_return, min_return)
        return episode_return, episode_return_nor, failure



def evaluate_episode_rtg_prom(
        env,
        state_dim,
        act_dim,
        model,
        max_length,
        target_cost=0.0001,
        deterministic=False,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        max_return=1,
        min_return=0,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    target_return_togo = target_return

    is_gym_env_v3 = env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3']
    if is_gym_env_v3:
        state = env.reset()
    else:
        state = env.reset()[0]
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)


    n_of_shots = 5
    group_state_prom = torch.zeros((n_of_shots, max_length, state_dim), device=device, dtype=torch.float32)
    group_act_prom = torch.zeros((n_of_shots, max_length, act_dim), device=device, dtype=torch.float32)
    group_rtg_prom = torch.zeros((n_of_shots, max_length, 1), device=device, dtype=torch.float32)
    group_log_p_pi = torch.zeros((n_of_shots, 1), device=device, dtype=torch.float32)

    for i in range(n_of_shots):
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = target_return_togo
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)


        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            deterministic,
        )
        actions[-1] = action

        states_norm = (states - state_mean) / state_std

        states_s0 = torch.unsqueeze(states_norm, 1)
        actions_a0 = torch.unsqueeze(actions, 1)
        rtg0 = torch.unsqueeze(target_return, 1)

        input_ids1 = dict()
        input_ids1.update(states=states_s0)
        input_ids1.update(actions=actions_a0)
        input_ids1.update(returns_to_go=rtg0)

        prom_out = model.generate(input_ids1)

        states_prom = prom_out['states'][:,:-1,:]
        actions_prom = prom_out['actions'][:,:-1,:]
        rtg_prom = prom_out['returns_to_go'][:,:-1,:]
        log_p_pi_prom = prom_out['log_p_pi'][:,:-1,:]
        min_log_p_pi = torch.min(log_p_pi_prom, 1)
        argmin_log_p_pi = min_log_p_pi[1][0][0].item()

        if argmin_log_p_pi + 1 < max_length:
            zero_pad_s = torch.zeros((max_length - argmin_log_p_pi - 1, states.shape[1]), 
                                     device=device, dtype=torch.float32)
            zero_pad_a = torch.zeros((max_length - argmin_log_p_pi - 1, actions.shape[1]), 
                                     device=device, dtype=torch.float32)
            zero_pad_r = torch.zeros((max_length - argmin_log_p_pi - 1, 1), 
                                     device=device, dtype=torch.float32)
            group_state_prom[i] = torch.cat([zero_pad_s, states_prom[0, :argmin_log_p_pi + 1, :]], dim=0)
            group_act_prom[i] = torch.cat([zero_pad_a, actions_prom[0, :argmin_log_p_pi + 1, :]], dim=0)
            group_rtg_prom[i] = torch.cat([zero_pad_r, rtg_prom[0, :argmin_log_p_pi + 1, :]])
        else:
            group_state_prom[i] = states_prom[0, argmin_log_p_pi - max_length + 1: argmin_log_p_pi + 1, :]
            group_act_prom[i] = actions_prom[0, argmin_log_p_pi - max_length + 1: argmin_log_p_pi + 1, :]
            group_rtg_prom[i] = rtg_prom[0, argmin_log_p_pi - max_length + 1: argmin_log_p_pi + 1, :]
        group_log_p_pi[i] = min_log_p_pi[0]

    best_n = torch.argmax(group_log_p_pi, dim=0)
    best_state_prom = group_state_prom[best_n.item()]
    best_act_prom = group_act_prom[best_n.item()]
    best_rtg_prom = group_rtg_prom[best_n.item()]

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    states = torch.cat([best_state_prom.reshape(-1, state_dim), states], dim=0)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    actions = torch.cat([best_act_prom.reshape(-1, act_dim), actions], dim=0)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.cat([best_rtg_prom.reshape(-1, 1), target_return], dim=0)

    episode_return, episode_length = 0, 0
    reward_que = deque()
    failure = False
    cost_epi = []
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            deterministic=deterministic,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if is_gym_env_v3:
            state, reward, done, info = env.step(action)
        else:
            state, reward, terminal, timeout, info = env.step(action)
            done = terminal or timeout

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        if 'cost' in info.keys():
            cost_epi.append(abs(info['cost']))

        if mode != 'delayed':
            pred_return = target_return[-1] - (reward/scale)
        else:
            pred_return = target_return[-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=0)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3']:
            reward_que.append(reward)
            if len(reward_que) > 40:
                _ = reward_que.popleft()

            if episode_length > 70:
                if sum(reward_que) < 0:
                    break

        if done:
            if 'cost' in info.keys():
                episode_cost = sum(cost_epi)
                cost_mean = episode_cost / len(cost_epi)
                if cost_mean > target_cost:
                    failure = True
                cost_epi = []
            if not is_gym_env_v3:
                if terminal:
                    failure = True
            break

    if env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3'] \
    and t+1 < max_ep_len:
        failure = True
    #print(failure, t+1, done, episode_return, ep_return*scale)

    if 'cost' in info.keys():
        return episode_return, episode_cost, failure
    else:
        episode_return_nor = normalize_returns(episode_return, max_return, min_return)
        return episode_return, episode_return_nor, failure
    


def evaluate_episode_rtg_prom2_parallel(
        env,
        state_dim,
        act_dim,
        model,
        target_cost=0.0001,
        rand=False,
        minmin=False,
        n_shots=5,
        om_len=3,
        len_of_prom=5,
        deterministic=False,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        max_return=1,
        min_return=0,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    target_return_togo = target_return

    is_gym_env_v3 = env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3']
    if is_gym_env_v3:
        state = env.reset()
    else:
        state = env.reset()[0]
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)


    n_of_shots = n_shots
    len_of_prom = len_of_prom
    group_state_prom = torch.zeros((n_of_shots, len_of_prom, state_dim), device=device, dtype=torch.float32)
    group_act_prom = torch.zeros((n_of_shots, len_of_prom, act_dim), device=device, dtype=torch.float32)
    group_rtg_prom = torch.zeros((n_of_shots, len_of_prom, 1), device=device, dtype=torch.float32)
    group_log_p_pi = torch.zeros((n_of_shots, 1), device=device, dtype=torch.float32)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return_togo
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)


    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

    if not rand:
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            deterministic,
        )
        actions[-1] = action

        states_norm = (states - state_mean) / state_std

        states_s0 = torch.unsqueeze(states_norm, 0)
        actions_a0 = torch.unsqueeze(actions, 0)
        rtg0 = torch.unsqueeze(target_return, 0)

        input_ids1 = dict()
        input_ids1.update(states=states_s0.detach())
        input_ids1.update(actions=actions_a0.detach())
        input_ids1.update(returns_to_go=rtg0.detach())
        
        from decision_transformer.models.decision_transformer import GenerationMode
        conf = {"output_mode": GenerationMode.GREEDY_SEARCH_WITH_OM_NO_PADDING}
        
        prom_output =[]
        for _ in range(n_of_shots):
            input_ids1 = dict()
            input_ids1.update(states=states_s0.detach())
            input_ids1.update(actions=actions_a0.detach())
            input_ids1.update(returns_to_go=rtg0.detach())
            output = model.generate(input_ids1, generation_config=conf, max_k=om_len)
            prom_output.append(output)
        

        states_prom = [prom_out['states'][:,:-1,:][0] for prom_out in prom_output]
        actions_prom = [prom_out['actions'][:,:-1,:][0] for prom_out in prom_output]
        rtg_prom = [prom_out['returns_to_go'][:,:-1,:][0] for prom_out in prom_output]
        min_om = torch.cat([torch.min(prom_out['occupancy_measure'], dim=1)[0] for prom_out in prom_output])
        argmin_om = torch.cat([torch.min(prom_out['occupancy_measure'], dim=1)[1] for prom_out in prom_output]).tolist()

        for i in range(n_of_shots):
            if argmin_om[i] + 1 < len_of_prom:
                zero_pad_s = torch.zeros((len_of_prom - argmin_om[i] - 1, state_dim), 
                                            device=device, dtype=torch.float32)
                zero_pad_a = torch.zeros((len_of_prom - argmin_om[i] - 1, act_dim), 
                                            device=device, dtype=torch.float32)
                zero_pad_r = torch.zeros((len_of_prom - argmin_om[i] - 1, 1), 
                                            device=device, dtype=torch.float32)
                group_state_prom[i] = torch.cat([zero_pad_s, states_prom[i][:argmin_om[i] + 1, :]], dim=0)
                group_act_prom[i] = torch.cat([zero_pad_a, actions_prom[i][:argmin_om[i] + 1, :]], dim=0)
                group_rtg_prom[i] = torch.cat([zero_pad_r, rtg_prom[i][:argmin_om[i] + 1, :]])
            else:
                group_state_prom[i] = states_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
                group_act_prom[i] = actions_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
                group_rtg_prom[i] = rtg_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]

        if not minmin:
            best_n = torch.argmax(min_om, dim=0).item()
        else:
            best_n = torch.argmin(min_om, dim=0).item()
        best_state_prom = group_state_prom[best_n]
        best_act_prom = group_act_prom[best_n]
        best_rtg_prom = group_rtg_prom[best_n]

        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        states = torch.cat([best_state_prom.reshape(-1, state_dim), states], dim=0)
        # actions = torch.from_numpy(action).reshape(1, act_dim).to(device=device, dtype=torch.float32)
        actions = torch.cat([best_act_prom.reshape(-1, act_dim), action.unsqueeze(0)], dim=0)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.cat([best_rtg_prom.reshape(-1, 1), target_return], dim=0)


        states_norm = (states - state_mean) / state_std
        
        states_s0 = torch.unsqueeze(states_norm, 0)
        actions_a0 = torch.unsqueeze(actions, 0)
        rtg0 = torch.unsqueeze(target_return, 0)

        input_ids1 = dict()
        input_ids1.update(states=states_s0.detach())
        input_ids1.update(actions=actions_a0.detach())
        input_ids1.update(returns_to_go=rtg0.detach())
        
        conf = {"output_mode": GenerationMode.GREEDY_SEARCH_WITH_TRUE_CLF}
        
        prom_output =[]
        for _ in range(n_of_shots):
            input_ids1 = dict()
            input_ids1.update(states=states_s0.detach())
            input_ids1.update(actions=actions_a0.detach())
            input_ids1.update(returns_to_go=rtg0.detach())
            output = model.generate(input_ids1, generation_config=conf, max_k=om_len)
            prom_output.append(output)

        states_prom = [prom_out['states'][:,:-1,:][0] for prom_out in prom_output]
        actions_prom = [prom_out['actions'][:,:-1,:][0] for prom_out in prom_output]
        rtg_prom = [prom_out['returns_to_go'][:,:-1,:][0] for prom_out in prom_output]
        min_om = torch.cat([torch.min(prom_out['occupancy_measure'], dim=1)[0] for prom_out in prom_output])
        argmin_om = torch.cat([torch.min(prom_out['occupancy_measure'], dim=1)[1] for prom_out in prom_output]).tolist()
        optimum_cond = torch.cat([prom_out['satisfy_condition'].view(1) for prom_out in prom_output])
        # argmax_optimum_cond = torch.cat([torch.max(prom_out['satisfy_condition'], dim=1)[1] for prom_out in prom_output]).tolist()


        for i in range(n_of_shots):
            if argmin_om[i] + 1 < len_of_prom:
                zero_pad_s = torch.zeros((len_of_prom - argmin_om[i] - 1, state_dim), 
                                            device=device, dtype=torch.float32)
                zero_pad_a = torch.zeros((len_of_prom - argmin_om[i] - 1, act_dim), 
                                            device=device, dtype=torch.float32)
                zero_pad_r = torch.zeros((len_of_prom - argmin_om[i] - 1, 1), 
                                            device=device, dtype=torch.float32)
                group_state_prom[i] = torch.cat([zero_pad_s, states_prom[i][:argmin_om[i] + 1, :]], dim=0)
                group_act_prom[i] = torch.cat([zero_pad_a, actions_prom[i][:argmin_om[i] + 1, :]], dim=0)
                group_rtg_prom[i] = torch.cat([zero_pad_r, rtg_prom[i][:argmin_om[i] + 1, :]])
            else:
                group_state_prom[i] = states_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
                group_act_prom[i] = actions_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
                group_rtg_prom[i] = rtg_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]

        
        best_n = torch.argmax(optimum_cond, dim=0).item()
        best_state_prom = group_state_prom[best_n]
        best_act_prom = group_act_prom[best_n]
        best_rtg_prom = group_rtg_prom[best_n]

        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        states = torch.cat([best_state_prom.reshape(-1, state_dim), states], dim=0)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        actions = torch.cat([best_act_prom.reshape(-1, act_dim), actions], dim=0)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.cat([best_rtg_prom.reshape(-1, 1), target_return], dim=0)


    else:
        for _ in range(500):
            if is_gym_env_v3:
                state, reward, _, _ = env.step(env.action_space.sample())
            else:
                state, reward, _, _, _ = env.step(env.action_space.sample())
            ep_return -= reward
        
        len_of_prom = 5
        rand_state = torch.zeros((len_of_prom, state_dim), device=device, dtype=torch.float32)
        rand_action = torch.zeros((len_of_prom, act_dim), device=device, dtype=torch.float32)
        rand_rtg = torch.zeros((len_of_prom, 1), device=device, dtype=torch.float32)
        for i in range(len_of_prom):
            rand_state[i] = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)

            action = env.action_space.sample()
            rand_action[i] = torch.from_numpy(action).reshape(1, act_dim).to(device=device, dtype=torch.float32)

            rand_rtg[i] = torch.tensor(ep_return).reshape(1, 1).to(device=device, dtype=torch.float32)

            if is_gym_env_v3:
                state, reward, done, info = env.step(action)
            else:
                state, reward, terminal, timeout, info = env.step(action)
                done = terminal or timeout
            
            ep_return -= reward

            for _ in range(50):
                if is_gym_env_v3:
                    state, reward, _, _ = env.step(env.action_space.sample())
                else:
                    state, reward, _, _, _ = env.step(env.action_space.sample())                
                ep_return -= reward

        if is_gym_env_v3:
            state = env.reset()
        else:
            state = env.reset(seed=random.randint(1000, 10000))[0]

        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        states = torch.cat([rand_state.reshape(-1, state_dim), states], dim=0)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        actions = torch.cat([rand_action.reshape(-1, act_dim), actions], dim=0)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.cat([rand_rtg.reshape(-1, 1), target_return], dim=0)


    episode_return, episode_length = 0, 0
    reward_que = deque()
    failure = False
    cost_epi = []
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            deterministic=deterministic,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if is_gym_env_v3:
            state, reward, done, info = env.step(action)
        else:
            state, reward, terminal, timeout, info = env.step(action)
            done = terminal or timeout

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        if 'cost' in info.keys():
            cost_epi.append(abs(info['cost']))

        if mode != 'delayed':
            pred_return = target_return[-1] - (reward/scale)
        else:
            pred_return = target_return[-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=0)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3']:
            reward_que.append(reward)
            if len(reward_que) > 40:
                _ = reward_que.popleft()

            if episode_length > 70:
                if sum(reward_que) < 0:
                    break

        if done:
            if 'cost' in info.keys():
                episode_cost = sum(cost_epi)
                cost_mean = episode_cost / len(cost_epi)
                if cost_mean > target_cost:
                    failure = True
                cost_epi = []
            if not is_gym_env_v3:
                if terminal:
                    failure = True
            break

    if env.unwrapped.spec.id in ['HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3'] \
    and t+1 < max_ep_len:
        failure = True

    if 'cost' in info.keys():
        episode_return_nor, episode_cost_nor = env.get_normalized_score(episode_return, episode_cost)
        return episode_return, episode_cost, failure, episode_return_nor, episode_cost_nor
    else:
        episode_return_nor = normalize_returns(episode_return, max_return, min_return)
        return episode_return, episode_return_nor, failure
    


