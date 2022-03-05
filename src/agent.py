import numpy as np
import torch
from .dqn import DQN, Episode, device


EPSILON_START = 0.8
EPSILON_MIN = 0.02
EPSILON_DECAY = 10**3


class Agent:
    def __init__(self, action_space: list) -> None:
        self.epsilon_start = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.step_counter = 0
        # action_space is news categories [0-17] = 18
        self.action_space = action_space
        self.action_count = {cat: 0 for cat in self.action_space}
        self.state, self.action, self.reward, self.next_state = None, None, None, None
        self.policy_net = DQN().to(device)

    def act(self, state: np.ndarray) -> str:
        # exploration = np.random.uniform(0, 1) < self.__get_epsilon__()
        # if exploration:
        #     action = self.action_space[np.random.randint(0, len(self.action_space))]
        # else:
        with torch.no_grad():
            prediction = self.policy_net(state)
            action_index = torch.argmax(prediction)
            action = self.action_space[action_index]
        self.action_count[action] += 1
        return action, prediction

    def get_episode(
        self,
        state: np.ndarray = None,
        action: str = None,
        reward: int = None,
        next_state: np.ndarray = None,
    ) -> tuple:
        self.state, self.action, self.reward, self.next_state = (
            state,
            action,
            reward,
            next_state,
        )
        episode = Episode(
            state,
            action,
            reward,
            next_state,
        )
        return episode

    def __get_epsilon__(self):
        epsilon = max(
            self.epsilon_min,
            self.epsilon_start - self.step_counter / self.epsilon_decay,
        )
        self.step_counter += 1
        return epsilon
