import pandas as pd
from ast import literal_eval

news = pd.concat(
    [
        pd.read_csv("dataset/train/news.tsv", sep="\t", header=None),
        pd.read_csv("dataset/dev/news.tsv", sep="\t", header=None),
        pd.read_csv("dataset/test/news.tsv", sep="\t", header=None),
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


def load_train_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    behaviors_train = pd.read_csv("dataset/train/behaviors.tsv", sep="\t", header=None)
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
    behaviors_dev = pd.read_csv("dataset/dev/behaviors.tsv", sep="\t", header=None)
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
    behaviors_test = pd.read_csv("dataset/test/behaviors.tsv", sep="\t", header=None)
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
    print(behaviors.head())
    print("news head:")
    print(news.head())


def save_dataset(df: pd.DataFrame, name: str) -> None:
    df.to_csv("dataset/" + name + ".tsv", sep="\t", index=False)
    print(name, "saved successfully!")


def load_featured_dataset() -> pd.DataFrame:
    """return behaviors, news"""
    behaviors = pd.read_csv(
        "dataset/behaviors.tsv",
        sep="\t",
        converters={"History": literal_eval, "Impression": literal_eval},
    )
    news = pd.read_csv("dataset/news.tsv", sep="\t")
    return behaviors, news


def load_news_dataset() -> pd.DataFrame:
    """return news"""
    return news
