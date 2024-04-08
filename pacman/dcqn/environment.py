import gymnasium as gym

env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = env.observation_space.shape
state_size = state_shape[0]
n_actions = env.action_space.n
