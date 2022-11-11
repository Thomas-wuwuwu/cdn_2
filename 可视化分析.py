## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
# from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
# % matplotlib inline

## 数据处理
from sklearn import preprocessing

## 数据降维处理的
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, SparsePCA

## 模型预测的
# import lightgbm as lgb
# import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

source_data = pd.read_csv('data/training_dataset.csv')
label = source_data['buffer_rate']
source_train_data = source_data.copy()
source_test_data = pd.read_csv('data/test_dataset.csv')
# 显示所有列
pd.set_option('display.max_columns', None)


def numplot():
    numerical_cols = source_train_data.select_dtypes(exclude='object').columns
    # print(numerical_cols)
    numeric_features = ['avg_fbt_time', 'synack1_ratio', 'tcp_conntime', 'icmp_lossrate',
                        'icmp_rtt', 'ratio_499_5xx', 'inner_network_droprate',
                        'inner_network_rtt', 'cpu_util', 'mem_util', 'io_await_avg',
                        'io_await_max', 'io_util_avg', 'io_util_max', 'ng_traf_level']

    plt.figure(figsize=(15, 15))
    i = 1

    for col in numeric_features:
        plt.subplot(5, 4, i)
        i += 1
        sns.distplot(source_train_data[col], label='train', color='r', hist=False)
        sns.distplot(source_test_data[col], label='test', color='b', hist=False)
    plt.tight_layout()
    plt.show()


def labelplot():
    ## 绘制标签的统计图，查看标签分布
    plt.figure(figsize=(12, 6))
    sns.distplot(label, bins=100)
    plt.show()

    plt.figure(figsize=(12, 6))
    label.plot.box()
    plt.show()


labelplot()
