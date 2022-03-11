import torch
import torch.optim as optim
import random
import mlflow
from src.agent import Agent
from src.environment import Environment
from src.load_dataset import load_featured_dataset
from src.dqn import ReplayMemory, DQN, optimize_model, device
from src.data_model import News


TARGET_UPDATE = 5
STOP = 10


class Model():
    def __init__(self) -> None:
        self.target_update = TARGET_UPDATE
        self.stop = STOP
        """load datasets"""
        self.B, self.N = load_featured_dataset()
        """create env, agent, memory & networks"""
        self.env = Environment(self.N, self.B)
        self.agent = Agent(self.env.get_action_space())
        self.memory = ReplayMemory()
        self.policy_net = self.agent.policy_net
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.responses = ["yes", "no"]
        self.recommended_news = News(news_id="", title="", abstarct="")


    def run_model(self, user_id: str="kasiffff") -> None:
        reward_cum_sum = 0
        iter_counter = 0
        with mlflow.start_run():
            while(True):
                import kasifffffff
                user_id =  kasifffffff.run_user_id
                print("RUN_USER_ID", user_id)
                state = self.env.get_state(user_id)
                action_category, action_tensor = self.agent.act(state)
                action_news_id = self.env.get_action_news_id(action_category)
                title, abstract = self.env.get_news_info(action_news_id)
                self.recommended_news.news_id = action_news_id
                self.recommended_news.title = title
                self.recommended_news.abstarct = abstract
                # print("News Title is:", title)
                # user_response = input("Do you like the news?(yes/no) ")
                user_response = random.choice(self.responses)
                if iter_counter == self.stop:
                    user_response = 'exit'
                if user_response == 'exit':
                    break
                reward = self.env.get_reward(user_response)
                next_state = self.env.update_state(
                    current_state=state, action=action_news_id, reward=reward, user_id=user_id
                )
                self.memory.push(
                    state, action_tensor, reward, next_state
                )
                # print("State is:", torch.mean(state).item())
                # print("Action is:", action_category)
                print("Reward is:", reward[0].item())
                # print("Next State is:", torch.mean(next_state).item())
                # print("Memory Size is:", memory.__len__())
                optimize_model(self.memory, self.policy_net, self.target_net, self.optimizer)
                if iter_counter % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                reward_cum_sum += reward[0].item()
                if iter_counter % 2 == 0:
                    mlflow.log_metric("reward cum sum", reward_cum_sum, step=int(iter_counter/2))
                if iter_counter % 4 == 0:
                    mlflow.sklearn.log_model(self.policy_net, 'test_model')
                # print("ITER", iter_counter, "\n")
                iter_counter += 1
                yield self.recommended_news

        print("---DONE---")


# if __name__ == "__main__":
#     user_id = "U687515"
#     run_model(user_id=user_id)