import os.path
import pandas as pd
import pickle
import catboost as cb
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(3047)


def valresult(df):
    result = pd.DataFrame(columns=['line_id', 'original', 'predict'])
    result['line_id'] = y_test_dataframe['line_id']
    result['original'] = y_test_dataframe['buffer_rate']
    result['predict'] = cat_pred_dataframe['predict']
    result.to_csv('result_1107_1.csv', index=None)


def evluation(y_true, y_pred):
    y = y_true['buffer_rate'].to_numpy()
    y_pre = y_pred['predict'].to_numpy()
    y_ave = np.mean(y)
    N = y.shape[0]
    MRE = np.sum(np.abs(y - y_pre) / y_ave) / N
    print(MRE)
    score = (1 - min(MRE, 1)) * 100
    print(score)


def Evaluation_Metrics(y_true, y_pred):
    mre = np.sum(np.abs(y_true - y_pred) / (y_true.shape[0])) / np.mean(y_true)
    score = (1 - min(1, mre)) * 100
    return score


source_data = pd.read_csv('data/training_dataset.csv')
label = source_data['buffer_rate']
# source_train_data = source_data.iloc[:, 2:17]
# source_train_data = source_train_data.drop(['inner_network_rtt', 'io_util_avg', 'io_util_max'], axis=1)
# source_train_data = source_data.drop(['buffer_rate'], axis=1)
source_train_data = source_data.copy()
source_test_data = pd.read_csv('data/test_dataset.csv')
features = source_test_data.columns
# 标签编码
colss = ['domain_name', 'node_name']
labs = {}
for col in colss:
    lab = LabelEncoder()
    lab.fit(source_train_data[col].values.tolist() + source_test_data[col].values.tolist())
    labs[col] = lab

for col in colss:
    source_train_data[col] = labs[col].transform(source_train_data[col])
    source_test_data[col] = labs[col].transform(source_test_data[col])

# 切分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(source_train_data, label, test_size=0.2, random_state=42,
                                                  shuffle=False)
print("原始样本集 X 大小为：", source_train_data.shape[0])
print("训练集 X_train 大小为：", X_train.shape[0])
print("测试集 X_test 大小为：", X_val.shape[0])
y_val_dataframe = pd.DataFrame(y_val)
y_val_dataframe['line_id'] = y_val_dataframe.index
y_val_dataframe.reset_index()

# cat_model = cb.CatBoostClassifier(iterations=3000,
#                                   depth=7,
#                                   learning_rate=0.01,
#                                   loss_function='MultiClass',
#                                   eval_metric='AUC',
#                                   logging_level='Verbose',
#                                   metric_period=50)

cat_model = cb.CatBoostRegressor(iterations=2000,
                                 learning_rate=0.05,
                                 depth=4,
                                 loss_function='RMSE',
                                 eval_metric='RMSE',
                                 random_seed=50,
                                 od_type='Iter',
                                 od_wait=50)

kfold = KFold(n_splits=5, shuffle=True, random_state=43)
test_predictions = np.zeros(len(source_test_data))
for fold, (train_idx, val_idx) in enumerate(kfold.split(source_train_data, source_train_data["buffer_rate"])):
    X_train, X_val = source_train_data[features].iloc[train_idx], source_train_data[features].iloc[val_idx]
    y_train, y_val = source_train_data["buffer_rate"].iloc[train_idx], source_train_data["buffer_rate"].iloc[val_idx]
    # model = cat(task_type="CPU")
    cat_model.fit(X_train, y_train, verbose=0)
    print(f"第{fold + 1}折测试集的精度:", Evaluation_Metrics(y_true=y_val, y_pred=cat_model.predict(X_val)))
    test_pred = cat_model.predict(source_test_data)
    test_predictions += test_pred / 5

# cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_model.save_model('save/1109_1', format='cbm')

# cat_pred = cat_model.predict(X_val)

# cat_pred_dataframe = pd.DataFrame(cat_pred, columns=['predict'], index=y_val_dataframe['line_id'])
# print(cat_pred_dataframe.head())

test_df = pd.DataFrame({'line_number': source_test_data.index+1, 'buffer_rate_prediction': test_predictions})
test_df.to_csv(f'result/test_1110_{5}fold_seed{43}_1.csv', index=False)