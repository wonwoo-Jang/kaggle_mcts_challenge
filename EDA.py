import os
import pandas as pd
import numpy as np

train_path = 'train.csv'
train_data = pd.read_csv(train_path)

print(f'train data shape: {train_data.shape}')
print(f'train data features: {train_data.columns}')


# features를 파일로 저장할거야 / feature에 해당하는 예시 값도 같이 저장할거야
num_samples = 2
sample = [train_data.iloc[i] for i in range(num_samples)]

for i in range(num_samples):
  with open(f'sample{i}.txt', 'w', encoding='utf-8') as file:
      for feature, value in sample[i].items():
          file.write(f'{feature}: {value}\n')

# constant인 features를 찾기
identical_features = train_data.columns[train_data.nunique() == 1]
with open(f'identical_features.txt', 'w', encoding='utf-8') as file:
  for feature in identical_features:
    file.write(f'{feature}\n')


save_eda_path = './EDA_source'
