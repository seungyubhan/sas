import gymnasium


def set_env(env_name, dataset):

    if env_name == 'hopper':
        env = gymnasium.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
        dataset_path = f'data/Hopper-{dataset}-v2.pkl'
    elif env_name == 'walker2d':
        env = gymnasium.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
        dataset_path = f'data/Walker2d-{dataset}-v2.pkl'
    elif env_name == 'humanoid':
        env = gymnasium.make('Humanoid-v2')
        max_ep_len = 1000
        env_targets = [6000]
        scale = 1000.
        dataset_path = f'data/humanoid-{dataset}-v2.pkl'
    elif env_name == 'dsrl_pointgoal1':
        env = gymnasium.make('OfflinePointGoal1Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [30]
        scale = 1.
        dataset_path = f'data/OfflinePointGoal1-{dataset}-v0.pkl'
    elif env_name == 'dsrl_pointpush1':
        env = gymnasium.make('OfflinePointPush1Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [15]
        scale = 1.
        dataset_path = f'data/OfflinePointPush1-{dataset}-v0.pkl'
    elif env_name == 'dsrl_pointbutton1':
        env = gymnasium.make('OfflinePointButton1Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [40]
        scale = 1.
        dataset_path = f'data/OfflinePointButton1-{dataset}-v0.pkl'
    elif env_name == 'dsrl_pointgoal2':
        env = gymnasium.make('OfflinePointGoal2Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [30]
        scale = 1.
        dataset_path = f'data/OfflinePointGoal2-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_pointpush2':
        env = gymnasium.make('OfflinePointPush2Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [12]
        scale = 1.
        dataset_path = f'data/OfflinePointPush2-{dataset}-v0.pkl'
    elif env_name == 'dsrl_pointbutton2':
        env = gymnasium.make('OfflinePointButton2Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [40]
        scale = 1.
        dataset_path = f'data/OfflinePointButton2-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_cargoal1':
        env = gymnasium.make('OfflineCarGoal1Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [40]
        scale = 1.
        dataset_path = f'data/OfflineCarGoal1-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_cargoal2':
        env = gymnasium.make('OfflineCarGoal2Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [30]
        scale = 1.
        dataset_path = f'data/OfflineCarGoal2-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_carpush1':
        env = gymnasium.make('OfflineCarPush1Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [16]
        scale = 1.
        dataset_path = f'data/OfflineCarPush1-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_carpush2':
        env = gymnasium.make('OfflineCarPush2Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [15]
        scale = 1.
        dataset_path = f'data/OfflineCarPush2-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_carbutton1':
        env = gymnasium.make('OfflineCarButton1Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [45]
        scale = 1.
        dataset_path = f'data/OfflineCarButton1-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_carbutton2':
        env = gymnasium.make('OfflineCarButton2Gymnasium-v0')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [42]
        scale = 1.
        dataset_path = f'data/OfflineCarButton2-{dataset}-v0.pkl'    
    elif env_name == 'dsrl_halfcheetah':
        env = gymnasium.make('OfflineHalfCheetahVelocityGymnasium-v1')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [3000]
        scale = 1000.
        dataset_path = f'data/OfflineHalfCheetahVelocity-{dataset}-v1.pkl'   
    elif env_name == 'dsrl_swimmer':
        env = gymnasium.make('OfflineSwimmerVelocityGymnasium-v1')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [250]
        scale = 100.
        dataset_path = f'data/OfflineSwimmerVelocity-{dataset}-v1.pkl'    
    elif env_name == 'dsrl_walker2d':
        env = gymnasium.make('OfflineWalker2dVelocityGymnasium-v1')
        env.continue_goal = False
        max_ep_len = 1000
        env_targets = [3500]
        scale = 1000.
        dataset_path = f'data/OfflineWalker2dVelocity-{dataset}-v1.pkl'    
    else:
        raise NotImplementedError


    return env, max_ep_len, env_targets, scale, dataset_path
