import base64
import glob
import io
import os
import random

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import gymnasium as gym
from IPython.core.display import HTML
from IPython.core.display_functions import display


# general architecture of the neural network
class Network(nn.Module):

    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# setting up the environment
env = gym.make('LunarLander-v2')
state_shape = env.observation_space.shape
state_size = state_shape[0]
n_actions = env.action_space.n

# initializing hyperparameters
lr = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3

# implementing experience replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if self.memory > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        xps = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([xp[0] for xp in xps if xp is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([xp[1] for xp in xps if xp is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([xp[2] for xp in xps if xp is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([xp[3] for xp in xps if xp is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([xp[4] for xp in xps if xp is not None]).astype(np.uint8)).float().to(self.device)

        return states, next_states, actions, rewards, dones


class Agent():

    def __init__(self, state_size, action_size):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > minibatch_size:
            xps = self.memory.sample(minibatch_size)
            self.learn(xps, discount_factor)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, xps, discount_factor):
        states, next_states, actions, rewards, dones = xps
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_model, target_model, interpolation_param):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(interpolation_param * local_param.data + (1.0 - interpolation_param) * target_param.data)

agent = Agent(state_size, action_size=n_actions)

n_episodes = 2_000
max_n_timesteps_per_episode = 1_000
epsilon_starting = 1.0
epsilon_ending = 0.01
epsilon_decay = 0.995
epsilon = epsilon_starting
scores_on_100_episodes = deque(maxlen=100)

for episode in range(1, n_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(max_n_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        if done:
            break

    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending, epsilon * epsilon_decay)

    print('\rEpisode {}\tAverage score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))

    if np.mean(scores_on_100_episodes) >= 200.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage score: {:.2f}'.format(episode - 100,
                                                                                     np.mean(scores_on_100_episodes)))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"


def show_video_of_model(agent: Agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)


show_video_of_model(agent, 'LunarLander-v2')


def show_video():
    video_list = glob.glob('*.mp4')
    if len(video_list) > 0:
        most_recent_vid = video_list[0]
        video = io.open(most_recent_vid, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(
            data='''<video alt="test" autoplay loop controls style="height:384px;"><source src="data:video/mp4;base64,{0}" /></video>'''.format(
                encoded.decode('ascii'))))
    else:
        print('Could not find video.')


show_video()