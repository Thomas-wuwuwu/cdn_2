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
std = StandardScaler()
X_fit = std.fit(X_train)
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_train_pre = rf.predict(X_train)
y_test_pre = rf.predict(X_test)
with open('save/randomforest.pickle', 'wb') as f:
    pickle.dump(rf, f)
rmse_rf = mean_squared_error(y_test, y_test_pre) ** (1 / 2)
print(rmse_rf)
r2_rf = r2_score(y_test, y_test_pre)
print(r2_rf)
