import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import string
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import requests
from load_dataset import load_train_datasets, get_dataset_info, save_dataset


"""read dataset"""
behaviors, news = load_train_datasets()
get_dataset_info(behaviors=behaviors, news=news)


"""checking news providers using regex"""
# def get_news_provider(df: pd.DataFrame):
#   # convert https://assets.msn.com/labs/mind/AAIKUGl.html to https://assets.msn.com/labs/mind/
#   provider = df['URL'].apply(lambda x: re.sub('[^\/]*$', '', x) if x is not np.NaN else np.NaN)
#   return provider
# provider = get_news_provider(news)
# print(provider.value_counts())


class Features:
    def __init__(self, news_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> None:
        # tfidf dimension reduction parameter
        self.tfidf_n_component = 64
        # get dataframes
        self.news_df = news_df
        self.behaviors_df = behaviors_df
        # dictionary defitions
        self.history_dict = self.__create_news_dict__(news_df)

    def fit(self, X, y=None) -> None:
        # does nothing
        X = X

    def transform(self, X) -> pd.DataFrame:
        """return behaviors dataframe, news dataframe"""
        self.behaviors_df = (
            self.behaviors_df.pipe(self.drop_na_history)
            .pipe(self.keep_unique_users)
            # .pipe(self.to_datetime)
            .pipe(self.to_list_history)
            .pipe(self.to_list_impression)
            .pipe(self.history_click_count)
            .pipe(self.to_dict_impression)
            .pipe(self.drop_behavior_cols)
        )

        self.news_df = self.__merg_dfs__(
            self.news_df, self.__dict_to_df__(self.history_dict, "History Click Count")
        )

        self.news_df = (
            self.news_df.pipe(self.news_popularity)
            .pipe(self.news_cat_one_hot)
            .pipe(self.news_subcat_one_hot)
            .pipe(self.news_title_tfidf)
            .pipe(self.drop_news_cols)
        )

        return self.behaviors_df, self.news_df

    def drop_na_history(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(subset=["History"], inplace=True)
        print("... done with drop_na_history() ...")
        return df

    def keep_unique_users(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop_duplicates(subset=["User ID"], keep="last", inplace=True, ignore_index=True)
        print("... done with keep_unique_users() ...")
        return df

    def to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Time"] = df["Time"].astype("datetime64[ns]")
        df.sort_values(by=["Time"], ignore_index=True, inplace=True)
        print("... done with to_datetime() ...")
        return df

    def to_list_history(self, df: pd.DataFrame) -> pd.DataFrame:
        df["History"] = df["History"].apply(lambda x: x[0 : len(x)].split(" "))
        print("... done with to_list_history() ...")
        return df

    def to_list_impression(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Impressions"] = df["Impressions"].apply(lambda x: x[0 : len(x)].split(" "))
        print("... done with to_list_impression() ...")
        return df

    def history_click_count(self, df: pd.DataFrame) -> pd.DataFrame:
        def __clicks_count__(news_ids: list) -> None:
            for news_id in news_ids:
                value = self.history_dict.get(news_id)
                value += 1
                self.history_dict[news_id] = value

        df["History"].apply(__clicks_count__)
        print("... done with history_click_count() ...")
        return df

    def to_dict_impression(self, df: pd.DataFrame) -> pd.DataFrame:
        def __keep_dict__(impression: list) -> dict:
            news = dict()
            for imp in impression:
                split = imp.split("-")
                news_id = split[0]
                news_value = split[1]
                news.update({news_id: news_value})
            return news

        df["Impressions"] = df["Impressions"].apply(__keep_dict__)
        print("... done with to_dict_impression() ...")
        return df

    def drop_behavior_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        print("... done with drop_behavior_cols() ...")
        return df.drop(["Impression ID"], axis=1)

    def news_popularity(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Popularity"] = df["History Click Count"]
        print("... done with news_popularity() ...")
        return df

    def news_cat_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        print("... done with news_cat_one_hot() ...")
        return pd.get_dummies(df, columns=["Category"])

    def news_subcat_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        print("... done with news_subcat_one_hot() ...")
        return pd.get_dummies(df, columns=["SubCategory"])

    def news_title_tfidf(self, df: pd.DataFrame) -> pd.DataFrame:
        def __preprocessor__(text: str) -> str:
            text = text.translate(str.maketrans(string.digits, " " * len(string.digits)))
            text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            return text

        def __tokenizer__(text: str) -> str:
            return word_tokenize(text)

        vectorizer = TfidfVectorizer(
            preprocessor=__preprocessor__,
            tokenizer=__tokenizer__,
            stop_words=stopwords.words("english"),
        )
        tfidf = pd.DataFrame.sparse.from_spmatrix(vectorizer.fit_transform(df["Title"]))
        t_svd = TruncatedSVD(n_components=self.tfidf_n_component)
        tfidf_svd = pd.DataFrame(t_svd.fit_transform(tfidf))
        print("... done with news_title_tfidf() ...")
        return pd.concat([df, tfidf_svd], axis=1)

    def drop_news_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        print("... done with drop_news_cols() ...")
        return df.drop(
            [
                "Title",
                "Abstract",
                "URL",
                "Title Entities",
                "Abstract Entities",
                "History Click Count",
            ],
            axis=1,
        )

    def __create_news_dict__(self, df: pd.DataFrame) -> pd.DataFrame:
        return {row: 0 for row in df["News ID"]}

    def __dict_to_df__(self, news_dict: dict, col_name: str) -> pd.DataFrame:
        df = pd.DataFrame(news_dict.values(), index=news_dict.keys()).rename(columns={0: col_name})
        df.index.names = ["News ID"]
        return df

    def __merg_dfs__(
        self, first_df: pd.DataFrame, second_df: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(first_df, second_df, on="News ID")


B = behaviors.copy()
N = news.copy()

estimators = [("features", Features(news_df=N, behaviors_df=B))]
pipe = Pipeline(estimators)
pipe.fit([N, B])
B, N = pipe.transform([N, B])

get_dataset_info(behaviors=B, news=N)

save_dataset(df=B, name="behaviors")
save_dataset(df=N, name="news")
