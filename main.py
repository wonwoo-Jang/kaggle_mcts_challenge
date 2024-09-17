from dataset import MCTSDataset
from model import TreeModel
from engine import Engine

import pandas as pd

train_path = 'train.csv'

X_train, X_valid, y_train, y_valid  = MCTSDataset(train_path).get_data(test_size=0.2, random_state=42)

print(f'features: {X_train.shape[1]}')
print(f'train samples: {X_train.shape[0]}')
print(f'valid samples: {X_valid.shape[0]}')


params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'lambda': 0,
    'alpha': 1
}

model = TreeModel(params)
engine = Engine(model, X_train, X_valid, y_train, y_valid)
print('train start')
engine.train()
print('training finish!\ntest start')
engine.val()

# feature_coeff_ = model.feature_selection()
# importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_coeff_})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# print(importance_df)
# importance_df.to_csv('selected_features.csv')