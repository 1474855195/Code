#导入SMAC包
import skopt
from skopt import forest_minimize
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from sklearn import metrics

# 读取文件
df1 = pd.read_csv(r'ant-1.3.csv', header=0)
df2 = pd.read_csv(r'ant-1.4.csv', header=0)
# 调参实验
x_train = df1.iloc[:, 0:19].values
y_train = df1.iloc[:, 20].values
x_test = df2.iloc[:, 0:19].values
y_true = df2.iloc[:, 20].values
# 定义model
m = RandomForestClassifier(random_state=1)
# 参数空间
from skopt.space import Real, Integer, Categorical

space = [Integer(10, 50, name="max_depth"),
         Categorical(('sqrt', 'log2'), transform='label', name="max_features"),
         Integer(1, 101, name='n_estimators')]
# 目标函数
from skopt.utils import use_named_args


@use_named_args(space)
def RF(**params):
    m.set_params(**params)
    roc_auc = sk_model_selection.cross_val_score(m, x_train, y_train, scoring='roc_auc', cv=10)
    return -roc_auc.mean()


result = forest_minimize(RF, space, n_calls=200, random_state=0, base_estimator='RF')
# 参数带回实验进行预测
model = RandomForestClassifier(random_state=1, max_depth=result.x[0], max_features=result.x[1],
                               n_estimators=result.x[2])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# 评分
probas = model.predict_proba(t_test)
fpr, tpr, thresholds = roc_curve(y_true, probas[:, 1])
auc = metrics.auc(fpr, tpr)