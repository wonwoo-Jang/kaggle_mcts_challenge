from dataset import MCTSDataset
from model import XGBModel, CatBoostModel, LightGBMModel
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
    param_grid_xgb = {
        'tree_method': ['hist'],
        'device': ['cuda'],
        'lambda': [0],
        'alpha': [1],
        'learning_rate': [0.1],
        'n_estimators': [1000],
        'max_depth': [6],
        'min_child_weight': [1],
    }

    param_grid_lgb = {
        'reg_lambda': [0],
        'reg_alpha': [1],
        'learning_rate': [0.01],
        'n_estimators': [500],
        'max_depth': [5],
        'min_child_weight': [1],
    }

    param_grid_cat = {
        'task_type': ['GPU'],
        'l2_leaf_reg': [0],
        'learning_rate': [0.1],
        'n_estimators': [1000],
        'depth': [6],
        'min_data_in_leaf': [1]
    }

    xgb_grid = ParameterGrid(param_grid_xgb)
    lgb_grid = ParameterGrid(param_grid_lgb)
    cat_grid = ParameterGrid(param_grid_cat)

    #NOTE: group k-fold validation
    group_kfold = GroupKFold(n_splits=opt.NUM_FOLD)

    for xbg_params, cat_param, lgb_param in zip(xgb_grid, cat_grid, lgb_grid):
        print(f'for xbg:\nlearning_rate: {xbg_params["learning_rate"]}, n_estimators: {xbg_params["n_estimators"]}, max_depth: {xbg_params["max_depth"]}, min_child_weight: {xbg_params["min_child_weight"]}')
        for model in ['lgb', 'cat', 'xgb']:
            print(f'model: {model}')
            rmse_train_avg,  rmse_val_avg= 0, 0
            for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(X, y, groups)):
                print(f'{fold+1}th fold')
                if model == 'xgb':
                    model = XGBModel(xbg_params)
                if model == 'lgb':
                    model = LightGBMModel(lgb_param)
                if model == 'cat':
                    model = CatBoostModel(cat_param)
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