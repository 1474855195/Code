import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from hyperopt import fmin,tpe,hp,partial,anneal,rand
from hyperopt.early_stop import no_progress_loss
from sklearn import metrics
import time

# 读取文件
df1 = pd.read_csv(r'ant-1.3.csv', header=0)
df2 = pd.read_csv(r'ant-1.4.csv', header=0)
x_train = df1.iloc[:, 0:19].values
y_train = df1.iloc[:, 20].values
x_test = df2.iloc[:, 0:19].values
y_true = df2.iloc[:, 20].values


def RF(args):
    RF = RandomForestClassifier(n_estimators=int(args['n_estimators']), max_depth=int(args['max_depth']),
                                max_features=args['max_features'], random_state=1)
    auc = sk_model_selection.cross_val_score(RF, x_train, y_train, scoring='roc_auc', cv=10)
    return -auc.mean()


# 参数空间
space = {'max_depth': hp.quniform('max_depth', 10, 50, 2),
         'max_features': hp.choice('max_features', ['sqrt', 'log2']),
         'n_estimators': hp.quniform('n_estimators', 1, 101, 5)}
# 代理函数
algo = partial(anneal.suggest)
time_start = time.process_time()
best = fmin(RF, space=space, algo=algo, max_evals=200, early_stop_fn=no_progress_loss(30))
time_end = time.process_time()
time_sum = time_end - time_start
# 设置离散变量
if best['max_features'] == 0:
    best['max_features'] = 'sqrt'
else:
    best['max_features'] = 'log2'
# 参数带回实验进行预测
model = RandomForestClassifier(n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']),
                               max_features=best['max_features'], random_state=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 评分
probas = model.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_true, probas[:, 1])
auc = metrics.auc(fpr, tpr)
