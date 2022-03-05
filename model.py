import numpy as np
import torch
import torch.optim as optim
import random
from src.agent import Agent
from src.environment import Environment
from src.dqn import DQN, ReplayMemory, optimize_model, device
from src.load_dataset import load_featured_dataset


TARGET_UPDATE = 5
N_ITER = 500


"""load datasets"""
B, N = load_featured_dataset()


"""create env, agent & memory"""
env = Environment(N, B)
agent = Agent(env.get_action_space())
memory = ReplayMemory()


policy_net = agent.policy_net
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())


l_users = ["U687515", "U192112", "U629430", "U449564", "U24161", "U79744", "U219005"]
l_response = ["yes", "no"]


"""specify user"""
user_id = input("Enter your ID: ")


def run():
    for iter in range(N_ITER):
        state = env.get_state(user_id)
        state_to_action = torch.from_numpy(state)
        action, prediction = agent.act(state_to_action.float())
        news_id = env.cat_based_news(action)
        title, abstract = env.get_news_info(news_id)
        print("News Title is:", title)
        print("News Abstract is:", abstract)
        # user_response = input("Do you like the news?(yes/no) ")
        user_response = random.choice(l_response)
        reward = env.get_reward(user_response)
        next_state = env.update_state(
            current_state=state, action=news_id, reward=reward, user_id=user_id
        )
        print("State is:", np.mean(state))
        print("Action is:", action)
        print("Reward is:", reward)
        print("Next State is:", np.mean(next_state))
        next_state = torch.from_numpy(np.vstack(next_state).astype(np.float))
        reward = torch.tensor([reward], device=device)
        memory.push(
            state_to_action.float(), prediction, reward, next_state.float()
        )
        print(memory.__len__())
        optimize_model(memory, policy_net, target_net, optimizer)
        if iter % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print("ITER", iter)
    print("---DONE---")


if __name__ == "__main__":
    run()
