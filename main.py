from dataset import MCTSDataset
from model import TreeModel
from engine import Engine

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GroupKFold
import pandas as pd
import argparse

if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-SEED', type=int, default=42)
    parser.add_argument('-DEVICE', type=str, default='cuda:0')
    parser.add_argument('-NUM_WORKERS', type=int, default=4)
    parser.add_argument('-BATCH_SIZE', type=int, default=8)
    parser.add_argument('-NUM_FOLD', type=int, default=5)
    parser.add_argument('-Tree-MODE', type=str, default='train')
    parser.add_argument('-CodeBert-MODE', type=str, default='train')
    opt = parser.parse_args()

    train_path = 'train.csv'

    X, y, groups = MCTSDataset(train_path).get_data()

    print(f'num features: {X.shape[1]}')
    print(f'num samples: {X.shape[0]}')
    
    #NOTE: hyper parameter tuning
    param_grid = {
        'tree_method': ['hist'],
        'device': ['cuda'],
        'lambda': [0],
        'alpha': [1],
        'learning_rate': [0.1],
        'n_estimators': [1000],
        'max_depth': [6],
        'min_child_weight': [1],
    }

    grid = ParameterGrid(param_grid)

    #NOTE: group k-fold validation
    group_kfold = GroupKFold(n_splits=opt.NUM_FOLD)

    for params in grid:
        print(f'learning_rate: {params["learning_rate"]}, n_estimators: {params["n_estimators"]}, max_depth: {params["max_depth"]}, min_child_weight: {params["min_child_weight"]}')
        rmse_train_avg,  rmse_val_avg= 0, 0
        for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(X, y, groups)):
            print(f'{fold+1}th fold')
            model = TreeModel(params)
            X_train, X_valid = X.iloc[train_idx].drop(columns=['GameRulesetName']), X.iloc[valid_idx].drop(columns=['GameRulesetName'])
            y_train, y_valid = y[train_idx], y[valid_idx]
            engine = Engine(model, X_train, X_valid, y_train, y_valid)
            rmse_train = engine.train()
            rmse_train_avg += rmse_train
            rmse_val = engine.val()
            rmse_val_avg += rmse_val
        print(f'maen rmse for train: {rmse_train_avg/opt.NUM_FOLD}, for valid: {rmse_val_avg/opt.NUM_FOLD}\n')

    # feature_coeff_ = model.feature_selection()
    # importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_coeff_})
    # importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # importance_df.to_csv('selected_features.csv')