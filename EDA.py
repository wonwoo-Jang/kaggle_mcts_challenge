import os
import pandas as pd

train_path = 'train.csv'
train_data = pd.read_csv(train_path)

save_eda_path = './EDA_source'

print(f'train data shape: {train_data.shape}')

# features를 파일로 저장할거야 / feature에 해당하는 예시 값도 같이 저장할거야
num_samples = 2
sample = [train_data.iloc[i] for i in range(num_samples)]

for i in range(num_samples):
  with open(os.path.join(save_eda_path, f'sample{i}.txt'), 'w', encoding='utf-8') as file:
    for feature, value in sample[i].items():
      file.write(f'{feature}: {value}\n')

# constant인 features를 찾기
identical_features = train_data.columns[train_data.nunique() == 1]
with open(os.path.join(save_eda_path, f'identical_features.txt'), 'w', encoding='utf-8') as file:
  for feature in identical_features:
    file.write(f'{feature}\n')

# null 값인 features 찾기 (null값 모든 샘플에 똑같이 있는 거 확인했음)
nan_features = train_data.columns[train_data.isna().all()].tolist()
with open(os.path.join(save_eda_path, f'nan_features.txt'), 'w', encoding='utf-8') as file:
  for feature in nan_features:
    file.write(f'{feature}\n')

# 의미있는 열 중에 string 타입인 features 찾기
val_features = train_data.drop(columns=identical_features)
val_features = val_features.drop(columns=nan_features)
string_columns = val_features.select_dtypes(include='object').columns.tolist()
print(f'string type columns: {string_columns}') # results: ['GameRulesetName', 'agent1', 'agent2', 'EnglishRules', 'LudRules']

# 게임 종류에 대해 탐색
num_game_types = train_data['GameRulesetName'].nunique()
num_lud_rules = train_data['LudRules'].nunique()
print(num_game_types, num_lud_rules) # results: 1377, 1373 / why? doesn't same game guarantee same rules?

game_rules_pairs = train_data[['GameRulesetName', 'LudRules']].drop_duplicates()
duplicates_rules = game_rules_pairs[game_rules_pairs.duplicated(subset=['LudRules'], keep=False)]
duplicates_rules.to_csv(os.path.join(save_eda_path, 'duplicated_names_with_same_rules.csv')) # NOTE: 다른 이름으로 같은 규칙이 중복되어 있는 것을 발견! 따라서 GameRulesetName 이 feature는 사용하지 말기!
