import torch
import torch.optim as optim
import random
from src.agent import Agent
from src.environment import Environment
from src.dqn import DQN, ReplayMemory, optimize_model, device
from src.load_dataset import load_featured_dataset


TARGET_UPDATE = 5
N_ITER = 10


"""load datasets"""
B, N = load_featured_dataset()


"""create env, agent, memory & networks"""
env = Environment(N, B)
agent = Agent(env.get_action_space())
memory = ReplayMemory()
policy_net = agent.policy_net
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

print(env.get_action_space())

"""specify user"""
# user_id = input("Enter your ID: ")
user_id = "U687515"
responses = ["yes", "no"]


def run_model():
    for iter in range(N_ITER):
        state = env.get_state(user_id)
        action_category, action_tensor = agent.act(state)
        action_news_id = env.get_action_news_id(action_category)
        title = env.get_news_info(action_news_id)
        # user_response = input("Do you like the news?(yes/no) ")
        user_response = random.choice(responses)
        reward = env.get_reward(user_response)
        next_state = env.update_state(
            current_state=state, action=action_news_id, reward=reward, user_id=user_id
        )
        memory.push(
            state, action_tensor, reward, next_state
        )
        print("News Title is:", title)
        print("State is:", torch.mean(state).item())
        print("Action is:", action_category)
        print("Reward is:", reward[0].item())
        print("Next State is:", torch.mean(next_state).item())
        print("Memory Size is:", memory.__len__())
        optimize_model(memory, policy_net, target_net, optimizer)
        if iter % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print("ITER", iter, "\n")
    print("---DONE---")


if __name__ == "__main__":
    run_model()
