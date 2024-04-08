import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import Tensor

from .hyper_params import lr, minibatch_size, discount_factor

from .preprocess import preprocess_frame
from .network import Network
from .memory import Memory


class Agent():
    def __init__(self, state_size, action_size):
        """
        memory, target q network, local q network, state size, action size, device (cuda), optimizer
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optim = optim.Adam(self.local_qnetwork.parameters(), lr=lr)
        self.memory = Memory(10_000)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.push((state, action, reward, next_state, done))
        if len(self.memory) > minibatch_size:
            xps = self.memory.sample(batch_size=minibatch_size)
            self.learn(xps, discount_factor)

    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, xps, discount_factor):
        # getting the data from experiences
        states, actions, rewards, next_states, dones = zip(*xps)

        # processing, turning data into good formats
        states = self.__torch_format(states).float().to(self.device)
        actions = self.__torch_format(actions).long().to(self.device)
        rewards = self.__torch_format(rewards).float().to(self.device)
        next_states = self.__torch_format(next_states).float().to(self.device)
        dones = self.__torch_format(dones, np.uint8).float().to(self.device)

        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def __torch_format(self, item, astype=None) -> Tensor:
        if astype:
            return torch.from_numpy(np.vstack(item).astype(astype))
        return torch.from_numpy(np.vstack(item))


