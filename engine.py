from sklearn.metrics import mean_squared_error

class Engine(object):
  def __init__(self, model, X_train, X_valid, y_train, y_valid):
    self.model = model
    self.X_train = X_train
    self.X_valid = X_valid
    self.y_train = y_train
    self.y_valid = y_valid

  def train(self):
    self.model.fit(self.X_train, self.y_train)
    self.y_pred = self.model.val(self.X_train)
    rmse = mean_squared_error(self.y_train, self.y_pred, squared=False)
    print(f"train RMSE: {rmse}")

  def val(self):
    self.y_pred = self.model.val(self.X_valid)
    rmse = mean_squared_error(self.y_valid, self.y_pred, squared=False)
    print(f"valid RMSE: {rmse}")