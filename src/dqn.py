from collections import deque, namedtuple
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 2
MEMORY_SIZE = 10000
INPUT_SIZE = 376  # state_size
OUTPUT_SIZE = 18  # action_space_size
HIDDEN_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Episode = namedtuple("Episode", ("state", "action", "reward", "next_state"))


class ReplayMemory:
    def __init__(self) -> None:
        self.capacity = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args: Episode) -> None:
        self.memory.append(Episode(*args))

    def sample(self) -> list:
        return sample(self.memory, self.batch_size)


class DQN(nn.Module):
    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


def optimize_model(
    memory: ReplayMemory, policy_net: DQN, target_net: DQN, optimizer: torch.optim
) -> None:
    if len(memory) < BATCH_SIZE:
        return
    episodes = memory.sample()
    batch = Episode(*zip(*episodes))
    state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, -1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).reshape(BATCH_SIZE, -1).expand(-1, OUTPUT_SIZE).reshape(-1)
    next_state_batch = torch.cat(batch.next_state).reshape(BATCH_SIZE, -1)
    state_action_values = (
        policy_net(state_batch)
        .gather(1, action_batch.type(torch.int64).unsqueeze(0))
        .squeeze(0)
    )
    next_state_values = target_net(next_state_batch).reshape(-1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()