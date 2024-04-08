from collections import deque

import torch
import numpy as np
from dcqn.environment import env, n_actions, state_size
from dcqn.agent import Agent


agent = Agent(state_size, action_size=n_actions)

n_eps = 2_000
max_n_timesteps_per_episode = 10_000
epsilon_starting = 1.0
epsilon_ending = 0.01
epsilon_decay = 0.995
epsilon = epsilon_starting
scores_on_100_eps = deque(maxlen=100)
solved_n_scores = 500

for ep in range(1, n_eps):
    state, _ = env.reset()
    score = 0
    for ts in range(max_n_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        if done:
            break

    scores_on_100_eps.append(score)
    epsilon = max(epsilon_ending, epsilon * epsilon_decay)

    print('\rEpisode {}\tAverage score: {:.2f}'.format(ep, np.mean(scores_on_100_eps)), end="")
    if ep % 100 == 0:
        print('\rEpisode {}\tAverage score: {:.2f}'.format(ep, np.mean(scores_on_100_eps)))

    if np.mean(scores_on_100_eps) >= solved_n_scores:
        print('\nEnvironment solved in {:d} episodes!\tAverage score: {:.2f}'.format(ep - 100, np.mean(scores_on_100_eps)))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break
