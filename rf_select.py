#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
#import pylab as plot

train = pd.read_csv('C1C2.csv')
train.head()

X = train[['A1', 'A2', 'A3','A4', 'B1', 'B2', 'B3', 'B4', 'B5','B6']]
y = train['C1']

print(X.std(axis=0))
print(y.std(axis=0))
print(X.mean(axis=0))

##数据标准化
#从sklearn.preprocessing导入数据标准化模块。
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理。
sX = ss_X.fit_transform(X)
sy = ss_y.fit_transform(y)
#sX=X
#sy=y
print(sX.std(axis=0))
print(sX.mean(axis=0))

from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归器LinearRegression。
lr = LinearRegression()
# 从sklearn导入特征筛选器。
from sklearn import feature_selection

#kfold=KFold(n_splits=3, shuffle=False, random_state=None)
#train random forest at a range of ensemble sizes in order to see how the mse changes
percentiles = range(10, 110, 10)
results = []
rf = RandomForestRegressor(n_estimators=240,max_features='sqrt',random_state=7)
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = i) #特征筛选
    X_train_fs = fs.fit_transform(sX,sy)
    print X_train_fs
    scores = -cross_val_score(rf, X_train_fs, sy, cv=10, scoring='neg_mean_squared_error')
    results = np.append(results, scores.mean())
print results

# 找到提现最佳性能的特征筛选的百分比。
opt = np.where(results == results.max())[0]
print opt
#print 'Optimal number of features %d' %percentiles[opt]

import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()




