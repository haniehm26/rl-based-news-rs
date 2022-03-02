"""# Basic Models"""
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD

import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

click_rate_sort = N.sort_values(by=['Impression Counts'], ignore_index=True, ascending=False)
target_news = 'N105407'
click_rate_sort.head()

save_B = B.copy()

def in_impression(news_dict, news_id=target_news):
  if news_id in news_dict:
    return news_dict[news_id]
  return np.NaN

save_B[target_news] = save_B['Impressions Dict'].apply(in_impression)
save_B.dropna(inplace=True)
print(save_B.shape)
save_B.head()

def my_read_csv(my_file, df):
  i = 0
  flag = False
  result = dict()
  with open(my_file) as csv_file:
    for line in csv_file:
      line = line.strip()
      if i>0:
        if 'U' in line:
          features = ''
          split = line.split('\"')
          index = int(split[0].split(',')[0])
          if index in df.index.values:
            flag = True
          else:
            flag = False
        if flag:
          if '[' in line:
            features+=line.split('\"')[1]
            features+='  '
            key = int(line.split('\"')[0].split(',')[0])
            result[key] = None
          elif ']' in line:
            features+=line.split('\"')[0]
            result[key] = np.fromstring(features[1:-1], sep='  ',dtype=float)
          else:
            features+=line
            features+='  '
      i+=1
      if i%5000000==0:
        print('done with', i/1000000, 'M lines!')
  return result

# mean_f = my_read_csv('mean_f.csv', save_B)

def get_mean_f(index):
  return mean_f[index.name]

# save_B['mean_f'] = save_B.apply(get_mean_f, axis=1)
# save_B.head()

no_news_id_df = N.copy()
no_news_id_df.drop(['News ID'], inplace=True, axis=1)


# !!!!!!!! we should make this code faster
# try this: N.loc[news]
def find_news_features(news):
  array = np.zeros((3, 378))
  index = 0
  for news_id in news:
    value = no_news_id_df[N['News ID']==news_id].loc[:].values
    array[index: ] = value
    index += 1
  return np.mean(array, axis=0)

B['Last3'] = B['History List'].apply(lambda x: x[-3:])
B['mean_f'] = B['Last3'].apply(find_news_features)

B.head()

save_B.reset_index(drop=True, inplace=True)
print(save_B.shape)
save_B.head()

from sklearn.preprocessing import MinMaxScaler
df = save_B.copy()
df = pd.DataFrame(df['mean_f'].apply(pd.Series))
df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df)
df[target_news] = save_B[target_news]
df.head()

from sklearn.model_selection import train_test_split

y = df[target_news]
X = df.drop([target_news], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

print('y==0/y==1 on train', y_train.value_counts()[0]/y_train.value_counts()[1])
print('y==0 on train', y_train.value_counts()[0])
print('y==1 on train', y_train.value_counts()[1])
print('y==0/y==1 on test', y_test.value_counts()[0]/y_test.value_counts()[1])
print('y==0 on test', y_test.value_counts()[0])
print('y==1 on test', y_test.value_counts()[1])

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

mnb = MultinomialNB().fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

gnb = GaussianNB().fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# svm = SVC().fit(X_train, y_train)
# y_pred_svm = svm.predict(X_test)

lr = LogisticRegression(penalty="l1", solver="saga", tol=0.1).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestClassifier(max_depth=250).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

def print_scores(model, y_test, y_pred):
  y_pred = y_pred.astype(int)
  y_test = y_test.astype(int)
  print('--- result of', model, '---')
  print('acc:', accuracy_score(y_test, y_pred))
  print('f1:', f1_score(y_test, y_pred))
  print('precision:', precision_score(y_test, y_pred))
  print('recall:', recall_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred), '\n')

print_scores('MNB', y_test, y_pred_mnb)
print_scores('GNB', y_test, y_pred_gnb)
# print_scores('SVM', y_test, y_pred_svm)
print_scores('LR', y_test, y_pred_lr)
print_scores('RF', y_test, y_pred_rf)

"""# RL Model"""

categories = N.columns[4:4+18].values
pd.get_dummies(categories).idxmax(1)
N['Category'] = pd.get_dummies(N[categories]).idxmax(1)

def amghezi(x):
  return x.split('_')[1]

N['Category'] = N['Category'].apply(amghezi)

categories = [cat.split('_')[1] for cat in categories]
categories

print(N.shape)
N.head()

B.head()

# mean_f = my_read_csv('mean_f.csv', B)
def get_mean_f(index):
  return mean_f[index.name]
  
B['mean_f'] = B.apply(get_mean_f, axis=1)
B.drop(['History List', 'Impressions Dict'], axis=1, inplace=True)
B.head()

from scipy import spatial

class Environment():
  def __init__(self, news_df, behavior_df):
    self.news_df = news_df
    self.behavior_df = behavior_df

  # state is 'mean_f' + 'impression_time_bucketting' (at first try impression_time_bucketting is ignored)
  def get_state(self, user_id):
    return self.behavior_df[self.behavior_df['User ID']==user_id]['mean_f'].values[0]

  def create_action_space(self):
    action_space = list()
    for news_cat in categories:
      action_space.append(news_cat)
    return action_space

  # action = news id that we recommend to a user
  def get_reward(self, action, user_id):
    print('News ID is', action)
    print('Category is', self.news_df[self.news_df['News ID']==action]['Category'])
    recommended_news_vector = self.news_df[self.news_df['News ID']==action].values[0][1:-1]
    mean_ones = self.behavior_df[self.behavior_df['User ID']==user_id]['mean_ones']
    mean_zeros = self.behavior_df[self.behavior_df['User ID']==user_id]['mean_zeros']
    mean_ones_similarity = 1 - spatial.distance.cosine(mean_ones, recommended_news_vector)
    mean_zeros_similarity = 1 - spatial.distance.cosine(mean_zeros, recommended_news_vector)
    return mean_ones_similarity / mean_zeros_similarity # , mean_ones_similarity - mean_zeros_similarity

  def update_action_space(action, reward, action_space):
    action_space[action] = reward
    return action_space

  def update_state_space(current_state, action, reward):
    if reward==1:
      current_state.add(action)
      new_state = update(mean_f)
      current_state = new_state
      return new_state
    elif reward==-1:
      pass

  def final_act(self, pre_act):
    exploration = (np.random.uniform(0, 1) < self.epsilon)
    df = self.news_df[self.news_df['Category']==pre_act]
    if exploration:
      action = df.iloc[np.random.randint(0, df.shape[0])]
    else:
      df.sort_values(by=['Popularity'], ignore_index=True, inplace=True, ascending=False)
      action = df['News ID'][0:5].values[np.random.randint(0, 5)]
    self.final_action_count[action] += 1
    return action

class Agent():
  def __init__(self, action_space, news_ids):
    self.epsilon = 0.01
    # action_space is news categories [0-16] = 17
    self.action_space = action_space
    self.pre_action_count = {cat: 0 for cat in categories}
    self.final_action_count = {news_id: 0 for news_id in news_ids}

  def primitive_act(self, state):
    exploration = (np.random.uniform(0, 1) < self.epsilon)
    if exploration:
      pre_action = categories[np.random.randint(0, len(categories))]
    else:
      prediction = self.model.predict(state)
      pre_action = np.argmax(prediction)
    self.pre_action_count[pre_action] += 1
    return pre_action

  def generate_model(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(len(self.action_space)), activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(self.action_space, activation='linear'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=keras.metrics.top_k_categorical_accuracy)
    return model

"""## Phase I"""

from collections import deque

class Environment():
  def __init__(self, news_df, behavior_df):
    self.news_df = news_df
    self.behavior_df = behavior_df
    self.categories = categories
    self.news_rand_rate = 0.1

  def get_state(self, user_id):
    return self.behavior_df[self.behavior_df['User ID']==user_id]['mean_f'].values[0]

  def get_action_space(self):
    action_space = list()
    for news_cat in self.categories:
      action_space.append(news_cat)
    return action_space

  # action = news id that we recommend to a user
  def get_reward(self, user_input):
    # print('News ID is', action)
    if user_input == 'yes':
      return 1
    elif user_input == 'no':
      return -1

  # returns new state and updates state in behavior dataframe
  def update_state(self, current_state, action, reward, user_id):
    if reward == 1:
      action_vector = self.news_df[self.news_df['News ID']==action].values[0][1:-1]
      new_state = (current_state + action_vector) / 2
      # !!! WARNING !!! this line of code is not updated
      self.behavior_df[self.behavior_df['User ID']==user_id]['mean_f'].values[0] = new_state
      return new_state
    elif reward == -1:
      new_state = current_state
      # alternatively new_state = (current_state - action_vector) / 2
      return new_state

  def cat_based_news(self, action):
    random_news = (np.random.uniform(0, 1) < self.news_rand_rate)
    df = self.news_df[self.news_df['Category']==action]
    if random_news:
      result = df.iloc[np.random.randint(0, df.shape[0])]
    else:
      df.sort_values(by=['Popularity'], ignore_index=True, inplace=True, ascending=False)
      result = df['News ID'][0:5].values[np.random.randint(0, 5)]
    return result

  def get_news_info(self, news_id):
    # !!!!!! self.news_df is a df without news features
    news = self.news_df[self.news_df['News ID']==news_id]
    title = news['Title']
    abstract = news['Abstaract']
    return title, abstract


class Agent():
  def __init__(self, action_space):
    self.epsilon = 0.01
    # action_space is news categories [0-16] = 17
    self.action_space = action_space
    self.action_count = {cat: 0 for cat in self.action_space}
    self.memory_max_size = 1000
    self.memory = deque(maxlen=self.memory_max_size)
    self.state = None # it can be mean_f or user_id
    self.action = ''
    self.reward = 0

  def act(self, state):
    # exploration = (np.random.uniform(0, 1) < self.epsilon)
    # if exploration:
    action = self.action_space[np.random.randint(0, len(self.action_space))]
    # else:
    #   prediction = self.model.predict(state)
    #   action = np.argmax(prediction)
    self.action_count[action] += 1
    return action

  def get_episode(self, state, action, reward, next_state):
    self.state, self.action, self.reward, self.next_state = state, action, reward, next_state
    episode = (np.mean(self.state), self.action, self.reward, np.mean(self.next_state))
    return episode

  def update_memory(self, episode):
    self.memory.append(episode)
    return self.memory

def run():
  env = Environment(N, B)
  agent = Agent(env.get_action_space())

  user_id = input('Enter your ID: ')
  state = env.get_state(user_id)
  print('State is', np.mean(state))
  action = agent.act(state)
  print('News Category is', action)
  news_id = env.cat_based_news(action)
  print('News ID is', news_id)
  user_response = input('Do you like this news?(yes/no) ')
  reward = env.get_reward(user_response)
  print('Reward is', reward)
  next_state = env.update_state(current_state=state, action=news_id, reward=reward, user_id=user_id)
  print('Next State is', np.mean(new_state))
  episode = agent.get_episode(state, action, reward, next_state)
  print('Episode is', episode)
  memory = agent.update_memory(episode)
  print('Memory is', memory)
  print('---DONE---')

run()

run()

N[N['News ID']=='N112324']



"""## Phase II"""

