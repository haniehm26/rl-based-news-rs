import numpy as np

from collections import deque

EPSILON = 0.01
MEMORY_MAX_SIZE = 1000


class Agent:
    def __init__(self, action_space: list) -> None:
        self.epsilon = EPSILON
        # action_space is news categories [0-17] = 18
        self.action_space = action_space
        self.action_count = {cat: 0 for cat in self.action_space}
        self.memory = deque(maxlen=MEMORY_MAX_SIZE)
        self.state, self.action, self.reward, self.next_state = None, None, None, None

    def act(self, state: np.ndarray) -> str:
        action = self.action_space[np.random.randint(0, len(self.action_space))]
        self.action_count[action] += 1
        return action

    def get_episode(
        self, state: np.ndarray, action: str, reward: int, next_state: np.ndarray
    ) -> tuple:
        self.state, self.action, self.reward, self.next_state = (
            state,
            action,
            reward,
            next_state,
        )
        episode = (
            self.state,
            self.action,
            self.reward,
            self.next_state,
        )
        return episode

    def update_memory(self, episode: tuple) -> deque:
        self.memory.append(episode)
        return self.memory
