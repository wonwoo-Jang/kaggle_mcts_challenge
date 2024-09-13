from dataset import MCTSDataset
from model import TreeModel
from engine import Engine

train_path = 'train.csv'

X_train, X_valid, y_train, y_valid  = MCTSDataset(train_path).get_data(test_size=0.2, random_state=42)

print(f'features: {X_train.shape[1]}')
print(f'train samples: {X_train.shape[0]}')
print(f'valid samples: {X_valid.shape[0]}')


params = {
    'tree_method': 'hist',
    'device': 'cuda'
}

model = TreeModel(params)
engine = Engine(model, X_train, X_valid, y_train, y_valid)
print('train start')
engine.train()
print('training finish!\ntest start')
engine.val()