import numpy as np

from src.agent import Agent
from src.environment import Environment
from src.load_dataset import load_featured_dataset


"""load datasets"""
B, N = load_featured_dataset()


"""create env & agent"""
env = Environment(N, B)
agent = Agent(env.get_action_space())


"""specify user"""
user_id = input("Enter your ID: ")


def run():
    state = env.get_state(user_id)
    action = agent.act(state)
    news_id = env.cat_based_news(action)
    title, abstract = env.get_news_info(news_id)
    print("News Title is:", title)
    print("News Abstract is:", abstract)
    user_response = input("Do you like the news?(yes/no) ")
    reward = env.get_reward(user_response)
    next_state = env.update_state(
        current_state=state, action=news_id, reward=reward, user_id=user_id
    )
    episode = agent.get_episode(state, action, reward, next_state)
    print("State is:", np.mean(state))
    print("Action is:", action)
    print("Reward is:", reward)
    print("Next State is:", np.mean(next_state))
    memory = agent.update_memory(episode)
    print(len(memory))
    print("---DONE---")


if __name__ == "__main__":
    while True:
        run()
