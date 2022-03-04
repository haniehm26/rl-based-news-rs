import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

from src.load_dataset import load_featured_dataset, get_dataset_info, save_dataset


"""load datasets"""
B, N = load_featured_dataset()
get_dataset_info(behaviors=B, news=N)


"""some random news"""
target_news = "N45124"


def create_label(news_dict: dict, news_id: str = target_news) -> int:
    news_dict = eval(news_dict)
    if news_id in news_dict:
        return news_dict[news_id]
    return np.NaN


def find_news_features(news: list) -> np.ndarray:
    array = np.zeros((3, 376))
    index = 0
    for news_id in news:
        array[index:] = N.loc[lambda df: df["News ID"] == news_id].values[0][1:]
        index += 1
    return np.mean(array, axis=0)


def keep_last3(x: list) -> list:
    x = eval(X)
    return x[-3:]


print("target news is:", target_news)
print("popularity is", N[N["News ID"] == target_news]["Popularity"].values[0])
B["label"] = B["Impressions"].apply(create_label)
B.dropna(inplace=True)
print("dataset shape:", B.shape)

# B["last3"] = B["History"][0:1000].apply(keep_last3)
# B["mean_f"] = B["last3"][0:1000].apply(find_news_features)
# print(B.columns)
# B = B[["User ID", "mean_f"]]
# B.dropna(inplace=True)
# print(B.columns)
# print(B.head())
# save_dataset(df=B, name="mean_f")

mean_f = pd.read_csv("D:\\Rahnema\\News Reccomender System\\dataset\\mean_f.tsv", sep="\t")
mean_f["mean_f"] = mean_f["mean_f"].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=" "))

df = mean_f.copy()
df = pd.DataFrame(df["mean_f"].apply(pd.Series))
df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df)
df["label"] = mean_f.merge(B, on="User ID", how="left")["label"]
print(df.head())

y = df["label"]
X = df.drop(["label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

print("y==0/y==1 on train", y_train.value_counts()[0] / y_train.value_counts()[1])
print("y==0 on train", y_train.value_counts()[0])
print("y==1 on train", y_train.value_counts()[1])
print("y==0/y==1 on test", y_test.value_counts()[0] / y_test.value_counts()[1])
print("y==0 on test", y_test.value_counts()[0])
print("y==1 on test", y_test.value_counts()[1])

mnb = MultinomialNB().fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

gnb = GaussianNB().fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

svm = SVC().fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

lr = LogisticRegression(penalty="l1", solver="saga", tol=0.1).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestClassifier(max_depth=250).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


def print_scores(model, y_test, y_pred) -> None:
    y_pred = y_pred.astype(int)
    y_test = y_test.astype(int)
    print("--- result of", model, "---")
    print("acc:", accuracy_score(y_test, y_pred))
    print("f1:", f1_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred, zero_division=0))
    print("recall:", recall_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred), "\n")


print_scores("MNB", y_test, y_pred_mnb)
print_scores("GNB", y_test, y_pred_gnb)
print_scores("SVM", y_test, y_pred_svm)
print_scores("LR", y_test, y_pred_lr)
print_scores("RF", y_test, y_pred_rf)
