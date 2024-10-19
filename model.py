import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

class XGBModel(object):
  def __init__(self, hyperparameter):
    self.hyperparameter = hyperparameter
    self.model = xgb.XGBRegressor(**hyperparameter)

  def fit(self, X_train, y_train):
    self.model.fit(X_train, y_train)

  def val(self, X_valid):
    y_pred = self.model.predict(X_valid)

    return y_pred
  
  def feature_selection(self):
    feature_importance =self.model.feature_importances_
    
    return feature_importance

class CatBoostModel(object):
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter
        self.model = CatBoostRegressor(**hyperparameter, verbose=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def val(self, X_valid):
        y_pred = self.model.predict(X_valid)

        return y_pred

    def feature_selection(self):
        feature_importance = self.model.get_feature_importance()

        return feature_importance

class LightGBMModel(object):
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter
        self.model = lgb.LGBMRegressor(boosting_type='gbdt', device_type='gpu', **hyperparameter, verbosity=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def val(self, X_valid):
        y_pred = self.model.predict(X_valid)

        return y_pred

    def feature_selection(self):
        feature_importance = self.model.feature_importances_

        return feature_importance
