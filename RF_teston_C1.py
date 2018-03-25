#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
import pylab as plot

train = pd.read_csv('C1C2.csv')
print train.head()
test=pd.read_csv('test.csv')
print test.head()
X = train[['A1', 'A2', 'A3','A4', 'B1', 'B2', 'B3', 'B4', 'B5','B6']]
y = train['C1']
X_test = test[['A1', 'A2', 'A3','A4', 'B1', 'B2', 'B3', 'B4', 'B5','B6']]

print(X.std(axis=0))
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
tX = ss_X.fit_transform(X_test)
print(sX.std(axis=0))
print(sX.mean(axis=0))

print(sy.std(axis=0))
print(sy.mean(axis=0))

print(tX.std(axis=0))
print(tX.mean(axis=0))
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

#train random forest at a range of ensemble sizes in order to see how the mse changes
rf=RandomForestRegressor(n_estimators=240,max_features='sqrt',random_state=7)

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估。
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=30)
X_train_fs = fs.fit_transform(sX, sy)
rf.fit(X_train_fs, sy)
X_test_fs = fs.transform(tX)
print tX
print X_test_fs
#rf.fit(X_train_fs, y_train)
# Accumulate mse on test set
prediction = rf.predict(X_test_fs)

print prediction
print ss_y.inverse_transform(prediction)




# Plot feature importance
featureImportance = rf.feature_importances_
#print featureImportance


VNames = np.array(['B1', 'B2', 'B3'])
#print VNames
# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0]) + .5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, VNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()

#printed output
#MSE
#0.314125711509




