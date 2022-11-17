#导入SMAC包
import skopt
from skopt import forest_minimize
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from sklearn import metrics
import time

# 读取文件
df1 = pd.read_csv(r'ant-1.3.csv', header=0)
df2 = pd.read_csv(r'ant-1.4.csv', header=0)
# 调参实验
x_train = df1.iloc[:, 0:19].values
y_train = df1.iloc[:, 20].values
x_test = df2.iloc[:, 0:19].values
y_true = df2.iloc[:, 20].values

# 定义model
m = SVC(probability=True)
# 参数空间
from skopt.space import Real, Integer, Categorical

space = [Categorical(('sigmoid', 'poly', 'rbf'), name='kernel'),
         Real(1.0, 10.0, name='C'),
         Real(0.0, 10.0, name='coef0')]
# 目标函数
from skopt.utils import use_named_args


@use_named_args(space)
def SVM(**params):
    m.set_params(**params)
    roc_auc = sk_model_selection.cross_val_score(m, x_train, y_train, scoring='roc_auc', cv=5)
    return -roc_auc.mean()


time_start = time.process_time()
result = forest_minimize(SVM, space, n_calls=200, random_state=42, base_estimator='RF')
time_end = time.process_time()
time_sum = time_end - time_start
# 参数带回实验进行预测
model = SVC(probability=True, kernel=result.x[0], C=result.x[1], coef0=result.x[2])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# 评分
probas = model.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_true, probas[:, 1])
auc = metrics.auc(fpr, tpr)