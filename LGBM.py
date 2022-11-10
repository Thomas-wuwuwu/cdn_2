from lightgbm import LGBMRegressor
import pandas as pd
import pickle
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

source_data = pd.read_csv('data/training_dataset.csv')
label = source_data['buffer_rate']
source_train_data = source_data.iloc[:, 2:17]
# print(source_train_data.head(10))
# print(label.head(10))
X_train, X_test, y_train, y_test = train_test_split(source_train_data, label, test_size=0.25, random_state=1,
                                                    shuffle=False)
fit_parms = {'early_stopping_rounds': 10,
             'eval_metric': 'rmse',
             'eval_set': [(X_test, y_test)],
             'eval_names': ['valid'],
             'verbose': 100}
lgb = LGBMRegressor(max_depth=20, learning_rate=0.01, n_estimators=1000)
lgb.fit(X_train, y_train, **fit_parms)
y_pre = lgb.predict(X_test)
with open('save/lgbm.pickle', 'wb') as f:
    pickle.dump(lgb, f)
print(r2_score(y_test, y_pre))
