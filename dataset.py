import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from datasets import Dataset # hugging face library

class MCTSDataset(object):
  def __init__(self, train_path):
    self.train_path = train_path
    self.train_data = pd.read_csv(train_path)

  def __len__(self):
    return self.train_data.shape[0]

  def agent_to_feature(self, agent: str, type: int):
    agent = agent.split('-')
    algorithm_candidate = ['UCB1', 'UCB1GRAVE', 'ProgressiveHistory', 'UCB1Tuned']
    algorithm = agent[1]
    exploration_const_candidate = ['0.1', '0.6', '1.41421356237']
    exploration_const = agent[2]
    playout_candidate = ['Random200', 'MAST', 'NST']
    playout = agent[3]
    score_bound_candidate = ['true', 'false']
    score_bound = agent[4]
    
    if type == 0:
      out = algorithm_candidate.index(algorithm)
    elif type == 1:
      out = exploration_const_candidate.index(exploration_const)
    elif type == 2:
      out = playout_candidate.index(playout)
    else:
      out = score_bound_candidate.index(score_bound)

    return out 

  def get_data(self):
    # constant or null 값이 아닌 feature만 의미있음
    identical_features = self.train_data.columns[self.train_data.nunique() == 1]
    nan_features = self.train_data.columns[self.train_data.isna().all()].tolist()
    self.train_data = self.train_data.drop(columns=identical_features)
    self.train_data = self.train_data.drop(columns=nan_features)

    # drop useless features: can't use in test data
    self.train_data = self.train_data.drop(columns=['Id', 'num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1'])

    # string features to vector
    string_features = ['GameRulesetName', 'agent1', 'agent2', 'EnglishRules', 'LudRules']

    useless_string_features = ['EnglishRules'] # GameRulesetName은 train test split의 기준으로 이용할 것이다
    self.train_data = self.train_data.drop(columns=useless_string_features)
    
    new_features = ['algorithm','exploration_const', 'playout', 'score']
    for agent in ['agent1', 'agent2']:
      for type, feature in enumerate(new_features):
        self.train_data[f'{feature}_{agent}'] = self.train_data[agent].apply(lambda x: self.agent_to_feature(x, type))
    self.train_data = self.train_data.drop(columns=['agent1', 'agent2'])

    # NOTE:LudRules는 따로 다룰 예정
    self.train_data = self.train_data.drop(columns=['LudRules'])

    # split train & valid set
    gameset = self.train_data['GameRulesetName']
    X = self.train_data.drop(columns=['utility_agent1'])
    y = self.train_data['utility_agent1']

    return X, y, gameset

