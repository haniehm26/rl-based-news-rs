import numpy as np
import pandas as pd
from ast import literal_eval

# TRAIN_PATH = "D:\\Rahnema\\News Reccomender System\\dataset\\train\\"
# DEV_PATH = "D:\\Rahnema\\News Reccomender System\\dataset\\dev\\"
# TEST_PATH = "D:\\Rahnema\\News Reccomender System\\dataset\\test\\"
# SAVE_PATH = "D:\\Rahnema\\News Reccomender System\\dataset\\"

TRAIN_PATH = "C:\\Users\\ASUS\\Desktop\\Hanieh\\dataset\\train\\"
DEV_PATH = "C:\\Users\\ASUS\\Desktop\\Hanieh\\dataset\\dev\\"
TEST_PATH = "C:\\Users\\ASUS\\Desktop\\Hanieh\\dataset\\test\\"
SAVE_PATH = "C:\\Users\\ASUS\\Desktop\\Hanieh\\dataset\\"

news = pd.concat(
    [
        pd.read_csv(TRAIN_PATH + "news.tsv", sep="\t", header=None),
        pd.read_csv(DEV_PATH + "news.tsv", sep="\t", header=None),
        pd.read_csv(TEST_PATH + "news.tsv", sep="\t", header=None),
    ]
)

news.columns = [
    "News ID",
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "Title Entities",
    "Abstract Entities",
]

news.drop_duplicates(subset=["News ID"], keep="last", inplace=True, ignore_index=True)

# relation_embedding_train = pd.read_csv(TRAIN_PATH + 'relation_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# relation_embedding_dev = pd.read_csv(DEV_PATH + 'relation_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# relation_embedding_test = pd.read_csv(TEST_PATH + 'relation_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)

# entity_embedding_train = pd.read_csv(TRAIN_PATH + 'entity_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# entity_embedding_dev = pd.read_csv(DEV_PATH + 'entity_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# entity_embedding_test = pd.read_csv(TEST_PATH + 'entity_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)


def load_train_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    behaviors_train = pd.read_csv(TRAIN_PATH + "behaviors.tsv", sep="\t", header=None)
    behaviors_train.columns = [
        "Impression ID",
        "User ID",
        "Time",
        "History",
        "Impressions",
    ]
    return behaviors_train, news


def load_dev_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    behaviors_dev = pd.read_csv(DEV_PATH + "behaviors.tsv", sep="\t", header=None)
    behaviors_dev.columns = [
        "Impression ID",
        "User ID",
        "Time",
        "History",
        "Impressions",
    ]
    return behaviors_dev, news


def load_test_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    behaviors_test = pd.read_csv(TEST_PATH + "behaviors.tsv", sep="\t", header=None)
    behaviors_test.columns = [
        "Impression ID",
        "User ID",
        "Time",
        "History",
        "Impressions",
    ]
    return behaviors_test, news


def get_dataset_info(behaviors: pd.DataFrame, news: pd.DataFrame) -> None:
    """print shape & head of dataframs"""
    print("behaviors shape:")
    print(behaviors.shape)
    print("news shape:")
    print(news.shape)
    print("behaviors head:")
    print(behaviors.head(2))
    print("news head:")
    print(news.head(2))


def save_dataset(df: pd.DataFrame, name: str) -> None:
    df.to_csv(SAVE_PATH + name + ".tsv", sep="\t", index=False)
    print(name, "saved successfully!")


def load_featured_dataset() -> pd.DataFrame:
    """return behaviors, news"""
    behaviors = pd.read_csv(
        SAVE_PATH + "behaviors.tsv",
        sep="\t",
        # dtype={"Time": 'str'},
        converters={"History": literal_eval, "Impression": literal_eval},
        # parse_dates=["Time"]
    )
    news = pd.read_csv(SAVE_PATH + "news.tsv", sep="\t")
    return behaviors, news


def load_news_dataset() -> pd.DataFrame:
    """return news"""
    return news
