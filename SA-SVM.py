import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from hyperopt import fmin,tpe,hp,partial,anneal,rand
from sklearn import metrics
import time
#读取文件
df1=pd.read_csv(r'ant-1.3.csv',header=0)
df2=pd.read_csv(r'ant-1.4.csv',header=0)
#调参实验
x_train=df1.iloc[:,0:19].values
y_train=df1.iloc[:,20].values
x_test=df2.iloc[:,0:19].values
y_true=df2.iloc[:,20].values
def SVM(args):
    SVC_H=SVC(probability=True,kernel=args['kernel'],C=float(args['C']),coef0=float(args['coef0']))
    roc_auc=sk_model_selection.cross_val_score(SVC_H,x_train,y_train,scoring='roc_auc',cv=5)
    return -roc_auc.mean()
#参数空间
space={'kernel': hp.choice('kernel',['sigmoid','poly','rbf']),
        'C': hp.uniform('C',1.0,10.0),
        'coef0': hp.uniform('coef0',1.0,10.0)}
#代理函数
algo=partial(anneal.suggest)
time_start = time.process_time()
best = fmin(SVM,space=space,algo = algo,max_evals=200)
time_end = time.process_time()
time_sum = time_end - time_start
#设置离散变量
if best['kernel']==0:
    best['kernel']='sigmoid'
if best['kernel'] == 1:
    best['kernel'] = 'poly'
if best['kernel']==2:
    best['kernel']='rbf'
#参数带回实验进行预测
model=SVC(probability=True,kernel=best['kernel'],C=float(best['C']),coef0=float(best['coef0']))
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#评分
probas=model.predict_proba(x_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)
