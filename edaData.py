import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pickle
from sklearn.impute import SimpleImputer

source_data = pd.read_csv('../data/training_dataset.csv')
label = source_data['buffer_rate']
c_source_data = source_data.copy()
c_source_data['id_domain'] = pd.Categorical(c_source_data.domain_name).codes
c_source_data['id_node'] = pd.Categorical(c_source_data.node_name).codes
c_source_data = c_source_data.drop('buffer_rate', axis=1)
c_source_data['id'] = c_source_data.index
train_data = c_source_data.iloc[:, 2:19]

test_data = pd.read_csv('../data/test_dataset.csv')
c_test_data =test_data.copy()
c_test_data['id_domain'] = pd.Categorical(c_test_data.domain_name).codes
c_test_data['id_node'] = pd.Categorical(c_test_data.node_name).codes
c_test_data['id'] = c_test_data.index

# print(c_source_data.dtypes)


# 相关性的查看
# corr = source_data.corr()['buffer_rate']
def view_corr(df):
    "查看相关性"
    corr = df.corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr, cmap=plt.cm.RdYlBu_r, annot=True)
    print(corr)
    plt.show()


def missing_value_table(df):
    """
    计算缺失值
    """
    # 计算所有的缺失值
    mis_val = df.isnull().sum()
    # 百分比化
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # 合并
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_rename = mis_val_table.rename(columns={0: 'Missing values', 1: '% of total values'})
    # 删除完整的并排序
    # mis_val_rename = mis_val_rename[mis_val_rename.iloc[:, 1] != 0].sort_values('% of total values', ascending=False)
    return mis_val_rename


# 将其封装为
def features_add(df):
    "添加一些特征值"
    df['id'] = df.index
    poly_features = df[
        ['icmp_lossrate', 'icmp_rtt', 'mem_util', 'ratio_499_5xx', 'synack1_ratio', 'avg_fbt_time']]
    poly_transformer = PolynomialFeatures(degree=2)
    poly_transformer.fit(poly_features)
    poly_features = poly_transformer.transform(poly_features)
    print(poly_features.shape)
    after_features = poly_transformer.get_feature_names_out(
        input_features=['icmp_lossrate', 'icmp_rtt', 'mem_util', 'ratio_499_5xx', 'synack1_ratio', 'avg_fbt_time'])
    poly_features = pd.DataFrame(poly_features, columns=after_features)
    poly_features['id'] = df['id']
    train_df = df.merge(poly_features, on='id', how='left')
    return train_df


# train_poly = features_add(c_source_data)
# train_poly = train_poly.drop('id', axis=1)
# train_poly.to_pickle('../data/train_poly.pkl')
# print(train_poly.head(10))

test_poly = features_add(c_test_data)
test_poly = test_poly.drop('id', axis=1)
test_poly.to_pickle('../data/test_poly.pkl')
print(test_poly.head(10))