import torch.optim as optim
import mlflow
from src.agent import Agent
from src.environment import Environment
from src.load_dataset import load_featured_dataset
from src.dqn import ReplayMemory, DQN, optimize_model, device
from src.data_model import News


TARGET_UPDATE = 5
LOG_METRIC = 2
LOG_MODEL = 4


class Model:
    def __init__(self) -> None:
        """target update parameter"""
        self.target_update = TARGET_UPDATE

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

        self.user_id = None
        self.state = None
        self.action_tensor = None
        self.action_news_id = None

        self.iter_counter = 0
        self.reward_cum_sum = 0

        """start mlflow"""
        mlflow.start_run()

    def recommend_news(self, user_id: str) -> News:
        print("User ID is:", user_id)
        self.user_id = user_id
        self.state = self.env.get_state(user_id)
        action_category, self.action_tensor = self.agent.act(self.state)
        self.action_news_id = self.env.get_action_news_id(action_category)
        title, abstract = self.env.get_news_info(self.action_news_id)
        return News(news_id=self.action_news_id, title=title, abstarct=abstract)

    def get_user_response(self, user_response: int) -> None:
        print("User Response is:", user_response)
        reward = self.env.get_reward(user_response)
        next_state = self.env.update_state(
            current_state=self.state,
            action=self.action_news_id,
            reward=reward,
            user_id=self.user_id,
        )
        self.memory.push(self.state, self.action_tensor, reward, next_state)
        optimize_model(self.memory, self.policy_net, self.target_net, self.optimizer)
        if self.iter_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.reward_cum_sum += reward[0].item()
        if self.iter_counter % LOG_METRIC == 0:
            mlflow.log_metric(
                "reward cum-sum",
                self.reward_cum_sum,
                step=int(self.iter_counter / LOG_METRIC),
            )
        if self.iter_counter % LOG_MODEL == 0:
            mlflow.sklearn.log_model(self.policy_net, "test_model")
        print("Iteration:", self.iter_counter)
        self.iter_counter += 1
