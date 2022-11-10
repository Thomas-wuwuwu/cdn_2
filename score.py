import pandas as pd
import pickle
import catboost as cb
from cdn_2 import edaData
from sklearn.model_selection import train_test_split
import numpy as np
# result = pd.read_csv('result_1107_1.csv')
# y = result['original'].to_numpy()
# y_pre = result['predict'].to_numpy()
# y_ave = np.average(y_pre)
# print(y_ave)
# N = y.shape[0]
# print(N)
# MRE = np.sum(np.abs(np.subtract(y, y_pre)) / y_ave) / N
# print(MRE)
# score = (1 - min(MRE, 1)) * 100
# print(score)
model = cb.CatBoostRegressor()
model.load_model('../save/1105_1')
with open('data/test_poly.pkl', 'rb') as file:
    train2 = pickle.load(file)
source_test_data = train2.iloc[:, 2:]
cat_pred = model.predict(source_test_data)

data_frame = pd.DataFrame(cat_pred, columns=['buffer_rate_prediction'])
data_frame['line_number'] = data_frame.index+1
data_frame = data_frame[['line_number', 'buffer_rate_prediction']]
print(data_frame.head(10))
data_frame.to_csv('../result/test_1108_02.csv', index=None)