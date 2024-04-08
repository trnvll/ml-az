import random
from collections import deque

class Memory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, item):
        self.memory.append(item)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
