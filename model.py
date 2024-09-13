import xgboost as xgb

class TreeModel(object):
  def __init__(self, hyperparameter):
    self.hyperparameter = hyperparameter
    self.model = xgb.XGBRegressor(**hyperparameter)

  def fit(self, X_train, y_train):
    self.model.fit(X_train, y_train)

  def val(self, X_valid):
    y_pred = self.model.predict(X_valid)

    return y_pred