import pandas as pd
import pickle
import catboost as cb
from sklearn.model_selection import train_test_split
import numpy as np
source_data = pd.read_csv('../data/training_dataset.csv')
label = source_data['buffer_rate']
c_source_data = source_data.copy()
c_source_data['id_domain'] = pd.Categorical(c_source_data.domain_name).codes
c_source_data['id_node'] = pd.Categorical(c_source_data.node_name).codes
c_source_data = c_source_data.drop('buffer_rate', axis=1)
print(c_source_data.head())
with open('../data/train_poly.pkl', 'rb') as file:
    train2 = pickle.load(file)
source_train_data = train2.iloc[:, 2:]

X_train, X_test, y_train, y_test = train_test_split(source_train_data, label, test_size=0.2, random_state=42,
                                                    shuffle=False)
print("原始样本集 X 大小为：", source_train_data.shape[0])
print("训练集 X_train 大小为：", X_train.shape[0])
print("测试集 X_test 大小为：", X_test.shape[0])
y_test_dataframe = pd.DataFrame(y_test)
y_test_dataframe['line_id'] = y_test_dataframe.index
y_test_dataframe.reset_index()
print(y_test_dataframe.head())
# cat_model = cb.CatBoostClassifier(iterations=3000,
#                                   depth=7,
#                                   learning_rate=0.01,
#                                   loss_function='MultiClass',
#                                   eval_metric='AUC',
#                                   logging_level='Verbose',
#                                   metric_period=50)
cat_model = cb.CatBoostRegressor(iterations=5000,
                                 learning_rate=0.01,
                                 depth=5,
                                 loss_function='RMSE',
                                 eval_metric='RMSE',
                                 random_seed=50,
                                 od_type='Iter',
                                 od_wait=50)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
cat_model.save_model(fname='../save/1105_1', format='cbm')
cat_pred = cat_model.predict(X_test)
cat_pred_dataframe = pd.DataFrame(cat_pred, columns=['predict'], index=y_test_dataframe['line_id'])
result = pd.DataFrame(columns=['line_id', 'original', 'predict'])
result['line_id'] = y_test_dataframe['line_id']
result['original'] = y_test_dataframe['buffer_rate']
result['predict'] = cat_pred_dataframe['predict']
print(result.head())
# result.to_csv('result_1105_1.csv', index=None)
# result.plot()
y = result['original'].to_numpy()
y_pre = result['predict'].to_numpy()
y_ave = np.average(y_pre)
N = y.shape[0]
print(N)
MRE = np.sum(np.abs(np.subtract(y, y_pre)) / y_ave) / N
print(MRE)
score = (1 - min(MRE, 1)) * 100
print(score)