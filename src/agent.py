import numpy as np

from collections import deque

EPSILON = 0.01
MEMORY_MAX_SIZE = 1000


class Agent:
    def __init__(self, action_space: list) -> None:
        self.epsilon_start = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.step_counter = 0
        self.action_space = action_space
        self.action_count = {cat: 0 for cat in self.action_space}
        self.memory = deque(maxlen=MEMORY_MAX_SIZE)
        self.state, self.action, self.reward, self.next_state = None, None, None, None

    def act(self, state: torch.Tensor) -> str:
        exploration = np.random.uniform(0, 1) < self.__get_epsilon__()
        if exploration:
            action_tensor = torch.rand([(len(self.action_space))], device=device)
        else:
            with torch.no_grad():
                action_tensor = self.policy_net(state)
        action_index = torch.argmax(action_tensor)
        action_category = self.action_space[action_index]
        self.action_count[action_category] += 1
        return action_category, action_tensor

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
