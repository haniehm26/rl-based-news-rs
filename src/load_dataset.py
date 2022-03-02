from numpy import void
import pandas as pd

"""Load Datasets"""

TRAIN_PATH = 'D:\\Rahnema\\News Reccomender System\\dataset\\train\\'
DEV_PATH = 'D:\\Rahnema\\News Reccomender System\\dataset\\dev\\'
TEST_PATH = 'D:\\Rahnema\\News Reccomender System\\dataset\\test\\'

behaviors_train = pd.read_csv(TRAIN_PATH + 'behaviors.tsv', sep='\t', header=None)
behaviors_dev = pd.read_csv(DEV_PATH + 'behaviors.tsv', sep='\t', header=None)
behaviors_test = pd.read_csv(TEST_PATH + 'behaviors.tsv', sep='\t', header=None)

news = pd.concat([pd.read_csv(TRAIN_PATH + 'news.tsv', sep='\t', header=None), 
                  pd.read_csv(DEV_PATH + 'news.tsv', sep='\t', header=None),
                  pd.read_csv(TEST_PATH + 'news.tsv', sep='\t', header=None)])

news.columns=['News ID',
              'Category',
              'SubCategory',
              'Title',
              'Abstract',
              'URL',
              'Title Entities',
              'Abstract Entities']

behaviors_train.columns=['Impression ID',
                         'User ID',
                         'Time',
                         'History',
                         'Impressions']

news.drop_duplicates(subset=['News ID'], keep='last', inplace=True, ignore_index=True)

# relation_embedding_train = pd.read_csv(TRAIN_PATH + 'relation_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# relation_embedding_dev = pd.read_csv(DEV_PATH + 'relation_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# relation_embedding_test = pd.read_csv(TEST_PATH + 'relation_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)

# entity_embedding_train = pd.read_csv(TRAIN_PATH + 'entity_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# entity_embedding_dev = pd.read_csv(DEV_PATH + 'entity_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)
# entity_embedding_test = pd.read_csv(TEST_PATH + 'entity_embedding.vec', sep='\t', header=None).rename({0: 'ID'}, axis=1)

def get_train_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    return behaviors_train, news

def get_dev_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    return behaviors_dev, news

def get_test_datasets() -> pd.DataFrame:
    """return behaviors, news"""
    return behaviors_test, news

def get_dataset_info(behaviors: pd.DataFrame, news: pd.DataFrame) -> None:
    """print shape & columns of dataframs"""
    print('behaviors shape:')
    print(behaviors.shape)                   
    print('news shape:')
    print(news.shape)
    print('behaviors columns:')
    print(behaviors.columns)
    print('news columns:')
    print(news.columns)