import pandas as pd
import numpy as np
import joblib
df= pd.read_csv(r"C:\Users\harma\OneDrive\Desktop\datasets\stickerSales\df_cleaned6.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
X= df.drop(columns= ['id', 'num_sold'])
y= df['num_sold']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)
xgb_regression= xgb.XGBRegressor(objective= 'reg:squarederror', n_estimators= 300, learning_rate= 0.2, max_depth= 6, subsample= 0.8, random_state= 42)
xgb_regression.fit(X_train, y_train)
y_train_pred= xgb_regression.predict(X_train)
y_test_pred= xgb_regression.predict(X_test)
def evaluate_model(y_true, y_pred, dataset_type):
    mae= mean_absolute_error(y_true, y_pred)
    mse= mean_squared_error(y_true, y_pred)
    rmse= np.sqrt(mse)
    r2= r2_score(y_true, y_pred)
    print(f"{dataset_type} Evaluation")
    print(f"MAE: {mae: .2f}")
    print(f"MSE: {mse: .2f}")
    print(f"RMSE: {rmse: .2f}")
    print(f"R2 Score: {r2: .2f}")
    print()
evaluate_model(y_train, y_train_pred, "Training")
evaluate_model(y_test, y_test_pred, "Test")
joblib.dump(xgb_regression, "xgboost_sales_model.pkl")
print("successfully saved the model")
