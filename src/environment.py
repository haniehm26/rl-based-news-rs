import pandas as pd
import numpy as np

from .load_dataset import load_news_dataset


RANDOM_NEWS_RATE = 0.1
STATE_WINDOW = 3


class Environment:
    def __init__(self, news_df: pd.DataFrame, behavior_df: pd.DataFrame) -> None:
        self.news_df = news_df
        self.behavior_df = behavior_df
        self.categories = self.__get_news_categories__()
        self.news_rand_rate = RANDOM_NEWS_RATE
        self.state_window = STATE_WINDOW

    def get_state(self, user_id: str) -> np.ndarray:
        history = self.behavior_df[self.behavior_df["User ID"] == user_id]["History"].values[0]
        last_k_news = history[-self.state_window :]
        array = np.zeros((self.state_window, 376))
        index = 0
        for news_id in last_k_news:
            array[index:] = self.news_df.loc[lambda df: df["News ID"] == news_id].values[0][1:-1]
            index += 1
        return np.mean(array, axis=0)

    def get_action_space(self) -> list:
        action_space = list()
        for news_cat in self.categories:
            action_space.append(news_cat)
        return action_space

    def get_reward(self, user_input: str) -> int:
        if user_input == "yes":
            return 1
        elif user_input == "no":
            return -1

    # return new state and updates state in behavior dataframe
    def update_state(
        self, current_state: np.ndarray, action: str, reward: int, user_id: str
    ) -> np.ndarray:
        if reward == 1:
            action_vector = self.news_df[self.news_df["News ID"] == action].values[0][1:-1]
            new_state = (current_state + action_vector) / 2
            history = self.behavior_df[self.behavior_df["User ID"] == user_id]["History"].values[0]
            history.append(action)
            self.behavior_df[self.behavior_df["User ID"] == user_id]["History"].replace(history)
            return new_state
        elif reward == -1:
            new_state = current_state
            return new_state

    def cat_based_news(self, action: str) -> str:
        random_news = np.random.uniform(0, 1) < self.news_rand_rate
        df = self.news_df[self.news_df["Category"] == action]
        if random_news:
            result = df.iloc[np.random.randint(0, df.shape[0])]
        else:
            df = df.sort_values(by=["Popularity"], ignore_index=True, ascending=False)
            n_rows = df.shape[0]
            if n_rows < 5:
                n_news = n_rows
            else:
                n_news = 5
            result = df["News ID"][0:n_news].values[np.random.randint(0, n_news)]
        return result

    def get_news_info(self, news_id: str) -> str:
        NEWS_DATASET = load_news_dataset()
        news = NEWS_DATASET[NEWS_DATASET["News ID"] == news_id]
        title = news["Title"]
        abstract = news["Abstract"]
        return title.values[0], abstract.values[0]

    def __get_news_categories__(self) -> list:
        categories = self.news_df.columns[2 : 2 + 18].values
        pd.get_dummies(categories).idxmax(1)
        self.news_df["Category"] = pd.get_dummies(self.news_df[categories]).idxmax(1)
        self.news_df["Category"] = self.news_df["Category"].apply(lambda x: x.split("_")[1])
        categories = [cat.split("_")[1] for cat in categories]
        return categories
