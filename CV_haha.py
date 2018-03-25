#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
import pylab as plot

train = pd.read_csv('C1C2.csv')
train.head()

X = train[['A1', 'A2', 'A3','A4', 'B1', 'B2', 'B3', 'B4', 'B5','B6']]
y = train['C1']

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
print(sX.std(axis=0))
print(sX.mean(axis=0))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer,r2_score,mean_absolute_error
# 从sklearn.linear_model导入LinearRegression。
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器LinearRegression。
lr = LinearRegression()
# 使用训练数据进行参数估计。
scores1 = cross_val_score(lr, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores1
scores1=scores1.mean()

# 从sklearn.svm中导入支持向量机（回归）模型。
from sklearn.svm import SVR

linear_svr = SVR(kernel='linear')
scores2 = cross_val_score(linear_svr, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores2
scores2=scores2.mean()

poly_svr = SVR(kernel='poly')
scores3 = cross_val_score(poly_svr, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores3
scores3=scores3.mean()

rbf_svr = SVR(kernel='rbf',C=1, epsilon=0.19)
scores4 = cross_val_score(rbf_svr, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores4
scores4=scores4.mean()

# 从sklearn.neighbors导入KNeighborRegressor（K近邻回归器）。
from sklearn.neighbors import KNeighborsRegressor
uni_knr = KNeighborsRegressor(weights='uniform')
scores5 = cross_val_score(uni_knr, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores5
scores5=scores5.mean()

from sklearn.linear_model import ARDRegression
ard=ARDRegression()
scores6 = cross_val_score(ard, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores6
scores6=scores6.mean()

# 从sklearn.linear_model中导入Lasso。
from sklearn.linear_model import Lasso
# 从使用默认配置初始化Lasso。
lasso = Lasso(alpha=0.07)
scores7 = cross_val_score(lasso, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores7
scores7=scores7.mean()

# 从sklearn.linear_model导入Ridge。
from sklearn.linear_model import Ridge
# 使用默认配置初始化Riedge。
ridge = Ridge(alpha=1)
scores8 = cross_val_score(ridge, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores8
scores8=scores8.mean()

from sklearn.linear_model import LassoLars
lars=LassoLars(alpha=0.009)
scores9 = cross_val_score(lars, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores9
scores9=scores9.mean()

from sklearn.linear_model import ElasticNetCV
elasticnet=ElasticNetCV(l1_ratio=0.13)
scores10 = cross_val_score(elasticnet, sX, sy, cv=10,scoring='neg_mean_squared_error')
print scores10
scores10=scores10.mean()

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# 从sklearn导入特征筛选器。
from sklearn import feature_selection

fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = 100)#30
X_train_fs = fs.fit_transform(sX,sy)

# 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量rfr_y_predict中。
rfr = RandomForestRegressor(n_estimators=240,max_features='sqrt',random_state=7)
scores11 = cross_val_score(rfr, X_train_fs, sy, cv=10,scoring='neg_mean_squared_error')
print scores11
scores11=scores11.mean()
print ss_y.inverse_transform(scores6)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中。
gbr = GradientBoostingRegressor(n_estimators=69,
                                max_depth=1,
                                learning_rate=0.256,
                                subsample=0.4,
                                loss='ls', max_features='auto', random_state=60)
fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = 100)#50
X_train_fs2 = fs.fit_transform(sX,sy)
scores12 = cross_val_score(gbr, X_train_fs2, sy, cv=10,scoring='neg_mean_squared_error')
print scores12
scores12=scores12.mean()

print -scores1,-scores2,-scores3,-scores4,-scores5,-scores6,-scores7,-scores8,-scores9,-scores10,-scores11,-scores12
data =pd.Series([-scores1,-scores2,-scores3,-scores4,-scores5,-scores6,-scores7,-scores8,-scores9,-scores10,-scores11,-scores12],
                index = ['linear','svr_linear','svr_poly','svr_rbf','kNN','ARD','Lasso','Ridge','LARS','ElasticNet','RF','GBR'])
data.plot(kind = 'barh',color='b',alpha = 0.7)
plot.xlabel('Mean Squared Error about C1')
plt.show()